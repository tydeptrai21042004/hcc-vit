#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os
import json

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    @staticmethod
    def _capture_trainable_state(model):
        """Copy only trainable parameters to CPU for best-checkpoint selection."""
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @staticmethod
    @torch.no_grad()
    def _restore_trainable_state(model, state):
        current = dict(model.named_parameters())
        missing = [name for name in state if name not in current]
        if missing:
            raise RuntimeError(f"Cannot restore missing trainable parameters: {missing[:5]}")
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))

    @staticmethod
    def _json_value(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {str(k): Trainer._json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Trainer._json_value(v) for v in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    def train_classifier(self, train_loader, val_loader, test_loader):
        """Train with validation checkpoint selection and evaluate test once.

        The test loader is intentionally excluded from the epoch loop.  The
        model state with the highest validation top-1 score is restored before
        the single final test evaluation.
        """
        self.model.eval()
        self.save_prompt(0)

        total_epoch = int(self.cfg.SOLVER.TOTAL_EPOCH)
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = float("-inf")
        best_state = None
        log_interval = max(1, int(self.cfg.SOLVER.LOG_EVERY_N))

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        patience = 0
        train_start = time.perf_counter()
        completed_epochs = 0

        for epoch in range(total_epoch):
            completed_epochs = epoch + 1
            losses.reset()
            batch_time.reset()
            data_time.reset()

            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            try:
                lr = self.scheduler.get_last_lr()[0]
            except AttributeError:
                lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            self.model.train()
            end = time.time()
            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    break

                X, targets = self.get_input(input_data)
                data_time.update(time.time() - end)
                train_loss, _ = self.forward_one_batch(X, targets, True)
                if train_loss == -1:
                    raise RuntimeError("Training stopped because a non-finite loss was encountered")

                losses.update(float(train_loss.item()), X.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()

                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1)
                        + seconds_per_batch * total_data * (total_epoch - epoch - 1)
                    ))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f}, ".format(
                            idx + 1, total_data, float(train_loss.item())
                        )
                        + "{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch, data_time.val, str(eta)
                        )
                        + "max mem: {:.1f} GB".format(gpu_mem_usage())
                    )

            logger.info(
                "Epoch {} / {}: avg data time: {:.2e}, avg batch time: {:.4f}, "
                "average train loss: {:.4f}".format(
                    epoch + 1, total_epoch, data_time.avg, batch_time.avg, losses.avg
                )
            )
            self.scheduler.step()
            self.model.eval()
            self.save_prompt(epoch + 1)

            self.evaluator.update_iteration(epoch)
            val_metrics = self.eval_classifier(
                val_loader, "val", save=(epoch == total_epoch - 1)
            )
            if val_metrics is None or "top1" not in val_metrics:
                raise RuntimeError("Validation evaluation did not return a top1 metric")
            current_metric = float(val_metrics["top1"])

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                best_state = self._capture_trainable_state(self.model)
                logger.info(
                    f"Best epoch {best_epoch}: validation top1={100.0 * best_metric:.3f}%"
                )
                patience = 0
            else:
                patience += 1

            if patience >= int(self.cfg.SOLVER.PATIENCE):
                logger.info("No validation improvement. Breaking out of loop.")
                break

        if best_state is None:
            # TOTAL_EPOCH=0 is permitted for evaluation-only workflows.
            best_state = self._capture_trainable_state(self.model)
            best_epoch = 0
            best_metric = float("nan")
        self._restore_trainable_state(self.model, best_state)
        self.model.eval()

        self.evaluator.update_iteration(-1)
        test_metrics = {}
        if test_loader is not None:
            test_metrics = self.eval_classifier(test_loader, "test", save=True) or {}

        total_train_time = time.perf_counter() - train_start
        total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        summary = {
            "seed": 0 if self.cfg.SEED is None else int(self.cfg.SEED),
            "best_epoch": int(best_epoch),
            "best_val_top1": float(best_metric),
            "best_val_acc1_percent": float(100.0 * best_metric),
            "completed_epochs": int(completed_epochs),
            "total_train_time_sec": float(total_train_time),
            "mean_epoch_time_sec": float(total_train_time / max(1, completed_epochs)),
            "trainable_parameters": int(trainable_parameters),
            "total_parameters": int(total_parameters),
            "test": self._json_value(test_metrics),
        }

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(self.cfg.OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        torch.save(self.evaluator.results, os.path.join(self.cfg.OUTPUT_DIR, "eval_results.pth"))
        if self.cfg.MODEL.SAVE_CKPT:
            torch.save(
                {
                    "seed": summary["seed"],
                    "best_epoch": best_epoch,
                    "best_val_top1": best_metric,
                    "trainable_state": best_state,
                },
                os.path.join(self.cfg.OUTPUT_DIR, "best_trainable_state.pth"),
            )
        return summary

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(float(loss.item()), X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        metrics = self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )
        metrics = {} if metrics is None else dict(metrics)
        metrics["loss"] = float(losses.avg)

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
        return metrics
