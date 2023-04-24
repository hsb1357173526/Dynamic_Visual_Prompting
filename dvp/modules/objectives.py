import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import numpy as np


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_snli_ve(pl_module, batch):
    infer = pl_module.infer(batch)

    if pl_module.model_name == 'BERT':
        snli_ve_logits = pl_module.snli_ve_classifier(infer["cls_feats"])
    elif pl_module.model_name == 'T5':
        snli_ve_logits = pl_module.snli_ve_classifier(infer["decoder_feats"])
    else:
        raise NotImplementedError("error in forwarding")

    snli_ve_labels = batch["answers"]
    snli_ve_labels = torch.tensor(snli_ve_labels).to(pl_module.device).long()
    snli_ve_loss = F.cross_entropy(snli_ve_logits, snli_ve_labels)

    kab_score = 0

    if pl_module.search_stage:
        if pl_module.kab_update_state:
            kab_score = accuracy(snli_ve_logits, snli_ve_labels)[0]
            kab_score = kab_score.cpu().numpy()[0]

    ret = {
        "snli_ve_loss": snli_ve_loss,
        "snli_ve_logits": snli_ve_logits,
        "snli_ve_labels": snli_ve_labels,
        'kab_score': kab_score,
    }

    phase = "train" if pl_module.training else "val"
    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_ve_loss")(ret["snli_ve_loss"])
        acc = getattr(pl_module, f"{phase}_snli_ve_accuracy")(
            ret["snli_ve_logits"], ret["snli_ve_labels"]
        )
        pl_module.log(f"snli_ve/{phase}/loss", loss)
        pl_module.log(f"snli_ve/{phase}/accuracy", acc)

    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_ve_accuracy")(
                ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
            )
            pl_module.log(f"snli_ve/dev/loss", dev_loss)
            pl_module.log(f"snli_ve/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_ve_accuracy")(
                ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
            )
            pl_module.log(f"snli_ve/test/loss", test_loss)
            pl_module.log(f"snli_ve/test/accuracy", test_acc)

    return ret


def compute_gqa(pl_module, batch):
    infer = pl_module.infer(batch)

    if pl_module.model_name == 'BERT':
        gqa_logits = pl_module.gqa_classifier(infer["cls_feats"])
    elif pl_module.model_name == 'T5':
        gqa_logits = pl_module.gqa_classifier(infer["decoder_feats"])
    else:
        raise NotImplementedError("error in forwarding")

    gqa_targets = torch.zeros(len(gqa_logits)).long().to(pl_module.device)

    gqa_labels = batch["gqa_labels"]

    for i, _label in enumerate(gqa_labels):
        for l in _label:
            gqa_targets[i] = l

    gqa_loss = F.cross_entropy(gqa_logits, gqa_targets)

    kab_score = 0

    if pl_module.search_stage:
        if pl_module.kab_update_state:
            kab_score = accuracy(gqa_logits, gqa_targets)[0]
            kab_score = kab_score.cpu().numpy()[0]


    ret = {
        "gqa_loss": gqa_loss,
        "gqa_logits": gqa_logits,
        "gqa_targets": gqa_targets,
        "gqa_labels": gqa_labels,
        'kab_score': kab_score,
    }

    phase = "train" if pl_module.training else "val"


    loss = getattr(pl_module, f"{phase}_gqa_loss")(ret["gqa_loss"])
    acc = getattr(pl_module, f"{phase}_gqa_accuracy")(
        ret["gqa_logits"], ret["gqa_targets"]
    )

    pl_module.log(f"gqa/{phase}/loss", loss)
    pl_module.log(f"gqa/{phase}/accuracy", acc)

    return ret


def calculate_vqa_score(pred,label):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    index_list = np.argmax(pred,axis=1).tolist()
    score = 0
    for i in range(len(index_list)):
        score += label[i][index_list[i]]
    score= score / len(index_list)
    return score


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch)
    if pl_module.model_name == 'BERT':
        vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    elif pl_module.model_name == 'T5':
        vqa_logits = pl_module.vqa_classifier(infer["decoder_feats"])
    else:
        raise NotImplementedError("error in forwarding")

    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )
    

    kab_score = 0
    
    if pl_module.search_stage:
        if pl_module.kab_update_state:
            kab_score = calculate_vqa_score(vqa_logits,vqa_targets) * 100.0

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
        'kab_score' : kab_score,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}



def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


