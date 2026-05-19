import argparse
import json
import netrc as netrc_module
import os

import numpy as np
import requests
import uproot
from sklearn.metrics import roc_auc_score, roc_curve

CLASSES = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
SIGNAL_CLASSES = ['Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']

TPR_TARGETS = {
    'Hbb': 0.50, 'Hcc': 0.50, 'Hgg': 0.50, 'H4q': 0.50,
    'Hqql': 0.99, 'Zqq': 0.50, 'Wqq': 0.50, 'Tbqq': 0.50, 'Tbl': 0.995
}

parser = argparse.ArgumentParser()
parser.add_argument('--pred-root', required=True)
parser.add_argument('--wandb-run-id', required=True, help='W&B run ID to resume (e.g. jlr1njrp)')
args = parser.parse_args()

print(f"Loading {args.pred_root}")
f = uproot.open(args.pred_root)
tree = f['Events']
branches = tree.arrays(library='np')

scores = np.stack([branches[f'score_label_{c}'] for c in CLASSES], axis=1)
labels = np.stack([branches[f'label_{c}'] for c in CLASSES], axis=1)

# Test accuracy: fraction of jets where top predicted class matches true class
pred_class = np.argmax(scores, axis=1)
true_class = np.argmax(labels, axis=1)
test_acc = (pred_class == true_class).mean() * 100
print(f"Test accuracy: {test_acc:.4f}%")

# OvO macro AUC
auc_ovo = roc_auc_score(labels, scores, multi_class='ovo', average='macro')
print(f"Overall pairwise AUC (OvO): {auc_ovo:.6f}")

# Per-class AUC
print("\nPer-class AUC (vs all others):")
auc_per_class = {}
for i, cls in enumerate(CLASSES):
    auc_per_class[cls] = roc_auc_score(labels[:, i], scores[:, i])
    print(f"  {cls:<8}: {auc_per_class[cls]:.6f}")

# Rejection rates vs QCD at per-class TPR target
print(f"\n{'Class':<10} {'TPR target':>10} {'TPR actual':>10} {'FPR':>10} {'Rejection':>12}")
print("-" * 55)
rejection_per_class = {}
score_qcd = branches['score_label_QCD']
for cls in SIGNAL_CLASSES:
    target_tpr = TPR_TARGETS[cls]
    sig_label = branches[f'label_{cls}']
    binary_label = (sig_label == 1).astype(int)
    score_sig = branches[f'score_label_{cls}']
    fpr, tpr, thresholds = roc_curve(binary_label, score_sig)
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(thresholds) - 1)
    threshold = thresholds[idx]
    tpr_actual = tpr[idx]
    qcd_mask = (branches['label_QCD'] == 1)
    fpr_actual = (score_sig[qcd_mask] >= threshold).mean()
    rejection = 1.0 / fpr_actual if fpr_actual > 0 else float('inf')
    rejection_per_class[cls] = rejection
    print(f"{cls:<10} {target_tpr:>10.1%} {tpr_actual:>10.1%} {fpr_actual:>10.4%} {rejection:>12.1f}")

# Log to W&B via direct GraphQL — avoids wandb.init() (hangs) and wandb.Api().run() (fails)
def get_api_key():
    key = os.environ.get('WANDB_API_KEY')
    if key:
        return key
    try:
        n = netrc_module.netrc()
        auth = n.authenticators('api.wandb.ai')
        if auth:
            return auth[2]
    except Exception:
        pass
    raise RuntimeError("No W&B API key found. Run: wandb login")

def graphql(query, variables, api_key):
    resp = requests.post(
        "https://api.wandb.ai/graphql",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

metrics = {"test/acc": test_acc, "test/auc_ovo_macro": auc_ovo}
for cls in CLASSES:
    metrics[f"test/auc_{cls}"] = auc_per_class[cls]
for cls in SIGNAL_CLASSES:
    tpr_label = int(TPR_TARGETS[cls] * 100)
    metrics[f"test/rejection_{cls}_at{tpr_label}eff"] = rejection_per_class[cls]

api_key = get_api_key()

# Step 1: resolve run name → internal bucket id
result = graphql("""
    query GetRun($entity: String!, $project: String!, $run: String!) {
        project(name: $project, entityName: $entity) {
            run(name: $run) { id name displayName }
        }
    }
""", {"entity": "martina-jorgensen-cern", "project": "par-t-quant", "run": args.wandb_run_id}, api_key)

run_data = result.get("data", {}).get("project", {}).get("run")
if run_data is None:
    print(f"\nRun '{args.wandb_run_id}' not found. Full response: {result}")
    raise SystemExit(1)
bucket_id = run_data["id"]
print(f"Found run: {run_data['displayName']} (name={run_data['name']}, id={bucket_id})")

# Step 2: write summary metrics
graphql("""
    mutation UpsertBucket($id: String!, $summaryMetrics: String) {
        upsertBucket(input: {id: $id, summaryMetrics: $summaryMetrics}) {
            bucket { id }
        }
    }
""", {"id": bucket_id, "summaryMetrics": json.dumps(metrics)}, api_key)

print("\nLogged to W&B successfully.")
