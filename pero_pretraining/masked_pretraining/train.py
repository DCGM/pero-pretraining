import os
import cv2
import json
import torch
import argparse

from functools import partial
from safe_gpu.safe_gpu import GPUOwner

from pero_pretraining.common.dataset import Dataset, DatasetLMDB
from pero_pretraining.common.helpers import get_checkpoint_path, get_visualization_path
from pero_pretraining.common.dataloader import create_dataloader, BatchCreator
from pero_pretraining.common.lr_scheduler import WarmupSchleduler

from pero_pretraining.masked_pretraining.model import init_backbone, init_head, MaskedTransformerEncoder
from pero_pretraining.masked_pretraining.tester import Tester
from pero_pretraining.masked_pretraining.trainer import Trainer
from pero_pretraining.masked_pretraining.visualizer import MaskedVisualizer as Visualizer
from pero_pretraining.masked_pretraining.batch_operator import BatchOperator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn-labels-file", help="Path to the training labels file.", required=True)
    parser.add_argument("--tst-labels-file", help="Path to the test labels file.")
    parser.add_argument("--lmdb-path", help="Path to the LMDB.", required=True)
    parser.add_argument("--augmentations", help="One of the predefined augmentations.", required=False, default=None)

    parser.add_argument("--batch-size", help="Batch size.", type=int, default=16)
    parser.add_argument("--learning-rate", help="Learning rate.", type=float, default=0.0002)
    parser.add_argument("--masking-prob", help="Masking probability.", type=float, default=0.15)
    parser.add_argument("--dropout", help="Dropout.", type=float, default=0.05)
    parser.add_argument("--start-iteration", help="Start iteration.", type=int, default=0)
    parser.add_argument("--end-iteration", help="End iteration.", type=int, default=100000)
    parser.add_argument("--max-line-width", help="Max line width.", type=int, default=2048, required=False)
    parser.add_argument("--warmup-iterations", help="Number of warmup iterations.", type=int, default=10000, required=False)
    parser.add_argument("--fill-width", help="Fill the maximum width with text lines (as long as they fit).", action="store_true")
    parser.add_argument("--exact-width", help="Fill the maximum width with text lines exactly (only effective with --fill-width).", action="store_true")

    parser.add_argument("--backbone", help="Backbone definition.", type=json.loads, default="{}")
    parser.add_argument("--head", help="Head definition.", type=json.loads, default="{}")

    parser.add_argument("--view-step", help="Number of iterations between testing.", type=int, default=500)
    parser.add_argument("--checkpoints", help="Path to a directory where checkpoints are saved.", default=None)
    parser.add_argument("--visualizations", help="Path to a directory where visualizations are saved.", default=None)

    args = parser.parse_args()
    return args


def init_model(device, backbone_definition, head_definition, path=None):
    backbone = init_backbone(backbone_definition)
    head = init_head(head_definition)

    model = MaskedTransformerEncoder(backbone, head)
    model.to(device)

    if path is not None:
        model.load(path)

    return model


def init_batch_operator(device, masking_prob):
    batch_operator = BatchOperator(device=device, masking_prob=masking_prob)
    return batch_operator


def init_datasets(trn_path, tst_path, lmdb_path, batch_size, augmentations, max_line_width, exact_width, fill_width):
    if "lmdb" in trn_path:
        trn_dataset = DatasetLMDB(lmdb_path=lmdb_path,
                                  lines_path=trn_path,
                                  augmentations=augmentations,
                                  pair_images=False,
                                  max_width=max_line_width,
                                  exact_width=exact_width,
                                  fill_width=fill_width)
    else:
        trn_dataset = Dataset(lmdb_path=lmdb_path,
                              lines_path=trn_path,
                              augmentations=augmentations,
                              pair_images=False,
                              max_width=max_line_width)

    if "lmdb" in tst_path:
        tst_dataset = DatasetLMDB(lmdb_path=lmdb_path,
                                  lines_path=tst_path,
                                  augmentations=None,
                                  pair_images=False,
                                  max_width=max_line_width,
                                  exact_width=exact_width,
                                  fill_width=fill_width)
    else:
        tst_dataset = Dataset(lmdb_path=lmdb_path,
                              lines_path=tst_path,
                              augmentations=None,
                              pair_images=False,
                              max_width=max_line_width)

    batch_creator = BatchCreator()

    trn_dataloader = create_dataloader(trn_dataset,
                                       batch_creator=batch_creator,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4)

    tst_dataloader = create_dataloader(tst_dataset,
                                       batch_creator=batch_creator,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=4)

    trn_dataloader.name = trn_dataset.name
    tst_dataloader.name = tst_dataset.name

    return trn_dataloader, tst_dataloader


def init_visualizers(batch_operator, model, trn_dataloader, tst_dataloader):
    trn_visualizer = Visualizer(batch_operator, model, trn_dataloader)
    tst_visualizer = Visualizer(batch_operator, model, tst_dataloader)

    return trn_visualizer, tst_visualizer


def init_testers(batch_operator, model, trn_dataloader, tst_dataloader):
    trn_tester = Tester(batch_operator, model, trn_dataloader, max_lines=1000)
    tst_tester = Tester(batch_operator, model, tst_dataloader)

    return trn_tester, tst_tester


def init_training(batch_operator, model, dataset, trn_tester, tst_tester, trn_visualizer, tst_visualizer, learning_rate,
                  warmup_iterations, checkpoints_directory, visualizations_directory):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = WarmupSchleduler(optimizer, learning_rate, warmup_iterations, 1)

    trainer = Trainer(batch_operator, model, dataset, optimizer, scheduler)
    trainer.on_view_step = partial(view_step_handler, 
                                   trn_tester=trn_tester, 
                                   tst_tester=tst_tester, 
                                   trn_visualizer=trn_visualizer, 
                                   tst_visualizer=tst_visualizer, 
                                   checkpoints_directory=checkpoints_directory, 
                                   visualizations_directory=visualizations_directory,
                                   scheduler=scheduler)

    return trainer


def init_directories(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def report(iteration, dataset, result, scheduler):
    errors_keys = sorted([key for key in result.keys() if key.startswith('errors_')],
                         key=lambda key: int(key.split('_')[-1]))

    print(f"TEST {dataset.name()} "
          f"iteration:{iteration} "
          f"loss:{result['loss']:.6f} "
          f"errors:{'|'.join(str(result[errors_key]) for errors_key in errors_keys)} "
          f"lr:{scheduler.current_lr:.6e}")


def test_model(iteration, tester, scheduler):
    result = tester.test()
    report(iteration, tester.dataloader, result, scheduler)


def save_model(model, path):
    model.save(path)


def visualize(visualizer, path):
    image = visualizer.visualize()
    cv2.imwrite(path, image)


def view_step_handler(iteration, model, elapsed_time, iteration_count, trn_tester, tst_tester, trn_visualizer,
                      tst_visualizer, checkpoints_directory, visualizations_directory, scheduler):
    print(f"Iteration: {iteration}, time: {elapsed_time:.2f} s, speed: {iteration_count / elapsed_time:.2f} it/s.")
    save_model(model, get_checkpoint_path(checkpoints_directory, iteration))

    test_model(iteration, trn_tester, scheduler)
    test_model(iteration, tst_tester, scheduler)

    visualize(trn_visualizer, get_visualization_path(visualizations_directory, iteration, "trn"))
    visualize(tst_visualizer, get_visualization_path(visualizations_directory, iteration, "tst"))


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = None
    if args.start_iteration > 0:
        checkpoint_path = get_checkpoint_path(args.checkpoints, args.start_iteration)

    model = init_model(device=device,
                       backbone_definition=args.backbone,
                       head_definition=args.head,
                       path=checkpoint_path)
    print(model)

    init_directories(args.checkpoints, args.visualizations)
    print("Directories initialized")

    batch_operator = init_batch_operator(device, masking_prob=args.masking_prob)
    print("Batch operator initialized")

    trn_dataset, tst_dataset = init_datasets(trn_path=args.trn_labels_file,
                                             tst_path=args.tst_labels_file,
                                             lmdb_path=args.lmdb_path,
                                             batch_size=args.batch_size,
                                             augmentations=args.augmentations,
                                             exact_width=args.exact_width,
                                             fill_width=args.fill_width,
                                             max_line_width=args.max_line_width)
    print("Datasets initialized")

    trn_visualizer, tst_visualizer = init_visualizers(batch_operator, model, trn_dataset, tst_dataset)
    print("Visualizers initialized")

    trn_tester, tst_tester = init_testers(batch_operator, model, trn_dataset, tst_dataset)
    print("Testers initialized")

    trainer = init_training(batch_operator=batch_operator,
                            model=model,
                            dataset=trn_dataset,
                            trn_tester=trn_tester,
                            tst_tester=tst_tester,
                            trn_visualizer=trn_visualizer,
                            tst_visualizer=tst_visualizer,
                            learning_rate=args.learning_rate,
                            warmup_iterations=args.warmup_iterations,
                            checkpoints_directory=args.checkpoints,
                            visualizations_directory=args.visualizations)
    print("Trainer initialized")

    trainer.train(start_iteration=args.start_iteration, end_iteration=args.end_iteration, view_step=args.view_step)
    print("Training finished")

    return 0


if __name__ == "__main__":
    gpu_owner = GPUOwner()
    exit(main())
