#!/usr/bin/env python3
#coding=utf8
import torch
import torch.nn.functional as F

def l1loss(pred, true):
    return F.l1_loss(pred, true)

def l2loss(pred, true):
    """ mean of loss^2 """
    return torch.mean(torch.pow(pred - true, 2))

def rmseloss(pred, true):
    """ squared mean of loss^2 """
    return torch.sqrt(l2loss(pred, true))

def nllloss(pred, true):
    return F.nll_loss(pred, true)

def klloss(pred, true):
    return F.kl_div(pred, true)

def lsadvloss(gen, true):
    """
    In discriminator case, arg should be reversed
    """
    return l2loss(gen, 1) / 2 + l2loss(true, 0) / 2

def calc_err(pred, true):
    ans = torch.max(pred,dim=1)[1]
    err = torch.mean((true!=ans).float())
    return err
