"""Une boucle d'époque explicite, commune à l'essai chronométré et à l'entraînement."""

from time import perf_counter

from sklearn.metrics import f1_score
import torch

from ianimaux.model import set_model_mode


def run_epoch(model, loader, criterion, optimizer=None, max_batches=None):
    training = optimizer is not None
    set_model_mode(model, training)
    total_loss = 0.0
    targets_all, predictions_all, rows = [], [], []
    start = perf_counter()
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            # CrossEntropyLoss expects raw logits, not softmax probabilities.
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Perte non finie : arrêter et vérifier les données/paramètres.")
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(targets)
        probabilities = logits.detach().softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        targets_all.extend(targets.tolist())
        predictions_all.extend(predictions.tolist())
        if not training:
            for target, prediction, scores in zip(targets.tolist(), predictions.tolist(), probabilities.tolist()):
                rows.append({"target": target, "prediction": prediction, "p_chat": scores[0], "p_chien": scores[1]})
    if not targets_all:
        raise ValueError("Aucune image traitée pendant l'époque.")
    metrics = {
        "loss": total_loss / len(targets_all),
        "accuracy": sum(truth == prediction for truth, prediction in zip(targets_all, predictions_all)) / len(targets_all),
        "f1_macro": float(f1_score(targets_all, predictions_all, labels=[0, 1], average="macro", zero_division=0)),
        "images": len(targets_all),
        "seconds": perf_counter() - start,
    }
    return metrics, rows
