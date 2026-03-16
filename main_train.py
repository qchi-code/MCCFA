#!/usr/bin/env python

import argparse
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR

from models.fewshot_anom import FewShotSeg
from dataloading.datasets import TrainDataset as TrainDataset
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_sv', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)

    # Training specs.
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--steps', default=50000, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--n_way', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--max_iterations', default=3000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_gamma', default=0.9, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--bg_wt', default=0.1, type=float)
    parser.add_argument('--t_loss_scaler', default=1.0, type=float)

    return parser.parse_args()


def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main():
    args = parse_arguments()

    # Fix random seed.
    setup_seed(args.seed)

    # Set up logging.
    logger = set_logger(args.save_root, 'train.log')
    logger.info(args)
    logger.info(f'Using fixed random seed: {args.seed}')

    # Setup the path to save.
    args.save_model_path = os.path.join(args.save_root, 'model.pth')

    # Init model.
    model = FewShotSeg()
    model = nn.DataParallel(model.cuda())

    # Init optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    milestones = [(ii + 1) * 1000 for ii in range(args.steps // 1000 - 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)

    # Define loss function.
    my_weight = torch.FloatTensor([args.bg_wt, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    # Define data set and loader.
    train_dataset = TrainDataset(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info('  Training on images not in test fold: ' +
                str([elem[len(args.data_root):] for elem in train_dataset.image_dirs]))

    # Start training.
    sub_epochs = args.steps // args.max_iterations
    logger.info('  Start training ...')
    start_time = time.time()

    for epoch in range(sub_epochs):
        batch_time, data_time, losses, q_loss, align_loss, prototype_contrast_loss = train(
            train_loader, model, criterion, optimizer, scheduler, args
        )

        logger.info('============== Epoch [{}] =============='.format(epoch))
        logger.info('  Batch time: {:6.3f}'.format(batch_time))
        logger.info('  Loading time: {:6.3f}'.format(data_time))
        logger.info('  Total Loss  : {:.5f}'.format(losses))
        logger.info('  Query Loss  : {:.5f}'.format(q_loss))
        logger.info('  Align Loss  : {:.5f}'.format(align_loss))
        logger.info('  contrast Loss  : {:.5f}'.format(prototype_contrast_loss))

    end_time = time.time()
    training_time = end_time - start_time
    logger.info('train time: {:.5f}'.format(training_time))

    # Save trained model.
    logger.info('  Saving model ...')
    torch.save(model.state_dict(), args.save_model_path)


def train(train_loader, model, criterion, optimizer, scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    q_loss = AverageMeter('Query loss', ':.4f')
    a_loss = AverageMeter('Align loss', ':.4f')
    contrast_loss = AverageMeter('contrast_loss loss', ':.4f')

    # Train mode.
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):

        # Extract episode data.
        support_images = [[shot.float().cuda() for shot in way]
                          for way in sample['support_images']]
        support_fg_mask = [[shot.float().cuda() for shot in way]
                           for way in sample['support_fg_labels']]

        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample['query_labels']], dim=0
        )

        # Log loading time.
        data_time.update(time.time() - end)

        # Compute outputs and losses.
        query_pred, align_loss, prototype_contrast_loss = model(
            support_images, support_fg_mask, query_images,
            train=True, t_loss_scaler=args.t_loss_scaler
        )

        query_loss = criterion(
            torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                  1 - torch.finfo(torch.float32).eps)),
            query_labels
        )

        loss = query_loss + align_loss + prototype_contrast_loss

        # Compute gradient and do SGD step.
        for param in model.parameters():
            param.grad = None

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss.
        losses.update(loss.item(), query_pred.size(0))
        q_loss.update(query_loss.item(), query_pred.size(0))
        a_loss.update(align_loss.item(), query_pred.size(0))
        contrast_loss.update(prototype_contrast_loss.item(), query_pred.size(0))

        # Log elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

    return batch_time.avg, data_time.avg, losses.avg, q_loss.avg, a_loss.avg, contrast_loss.avg


if __name__ == '__main__':
    main()