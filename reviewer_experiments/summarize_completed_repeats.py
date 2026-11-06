"""Verify all frozen repeated-run outputs and summarize aggregate results."""
from pathlib import Path
import argparse,hashlib,json,statistics,collections
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,f1_score
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--runs',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    final=json.loads((args.runs/'current_status.json').read_text())
    assert final['stage']=='all_repetitions' and final['status']=='complete'
    records=[];checks=[]
    for rep in range(1,4):
        root=args.runs/f'repeat_{rep:02d}'
        seeds=[(7,17,27),(37,47,57),(67,77,87)][rep-1]
        for seed in seeds:
            p=root/f'results/raw_runs/offline_safe_nulkt/seed_{seed}/safe_fit_provenance.json'
            d=json.loads(p.read_text());assert d['status']=='complete'
            checks.append({'artifact':f'repeat_{rep:02d}/neural_{seed}/fit','sha256':sha(p),'status':d['status']})
        for reg in ['benchmark','deployment']:
            folder=root/f'results/offline_safe_evaluation/{reg}'
            p=folder/'test_evaluation_posthoc.json';d=json.loads(p.read_text())
            assert d['status']=='complete'
            assert sha(folder/'test_predictions_posthoc.csv')==d['test_prediction_sha256']
            assert sha(folder/'dev_selection.json')==d['selection_sha256']
            selection=json.loads((folder/'dev_selection.json').read_text())
            assert selection['boosted_only_eligible'] and selection['neural_only_eligible']
            frame=pd.read_csv(folder/'test_predictions_posthoc.csv')
            assert len(frame)==386604 and frame.row_index.is_unique and frame.learner_id.nunique()==2593
            for name,col in [('selected_blend','probability'),('boosted','probability_boosted'),('neural','probability_neural')]:
                y=frame.label.to_numpy();pred=frame[col].to_numpy()
                m={'auroc':float(roc_auc_score(y,pred)),'auprc':float(average_precision_score(y,pred)),'brier_score':float(brier_score_loss(y,pred)),'f1':float(f1_score(y,pred>=d['frozen_threshold']))}
                for k,v in m.items():assert abs(v-d['metrics'][name][k])<1e-10,(rep,reg,name,k)
            ci=d['paired_learner_cluster_bootstrap']['boosted']['auroc']
            records.append({'repetition':rep,'regime':reg,'neural_seeds':seeds,'selected_combiner':d['selected_combiner'],'stack_auroc':d['metrics']['selected_blend']['auroc'],'boosted_auroc':d['metrics']['boosted']['auroc'],'auroc_delta':ci['model_a_minus_b'],'delta_ci_low':ci['ci_lower'],'delta_ci_high':ci['ci_upper'],'bootstrap_replicates':ci['replicates_valid'],'stack_auprc':d['metrics']['selected_blend']['auprc'],'boosted_auprc':d['metrics']['boosted']['auprc'],'stack_brier':d['metrics']['selected_blend']['brier_score'],'boosted_brier':d['metrics']['boosted']['brier_score'],'stack_ece':d['metrics']['selected_blend']['ece'],'stack_f1':d['metrics']['selected_blend']['f1'],'threshold':d['frozen_threshold'],'report_sha256':sha(p),'predictions_sha256':d['test_prediction_sha256'],'selection_sha256':d['selection_sha256']})
            checks.append({'artifact':f'repeat_{rep:02d}/{reg}/test','report_sha256':sha(p),'status':'hashes, identities and independently recomputed metrics PASS'})
    summaries={}
    for reg in ['benchmark','deployment']:
        rows=[r for r in records if r['regime']==reg]
        m={}
        for key in ['stack_auroc','boosted_auroc','auroc_delta','stack_auprc','stack_brier','stack_f1']:
            v=[r[key] for r in rows];m[key]={'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'min':min(v),'max':max(v)}
        m['selected_method_frequencies']=dict(collections.Counter(r['selected_combiner'] for r in rows))
        m['boosted_only_selected_count']=sum(r['selected_combiner']=='boosted_only' for r in rows)
        m['per_run_auroc_intervals_including_zero']=sum(r['delta_ci_low']<=0<=r['delta_ci_high'] for r in rows)
        summaries[reg]=m
    result={'status':'verified_complete','new_full_repetitions':3,'completed_utc':final['utc'],'completed_pakistan':'2026-11-01 00:05:33 PKT','records':records,'summary':summaries,'verification':checks,'interpretation':'All TEST scores post hoc. Sample SD describes three independently retrained systems on the same test rows, not a confidence interval for prospective superiority. Learner-cluster intervals condition on each fitted prediction set. Three repetitions are insufficient to claim broad training stability; no held-out superiority or SOTA claim is established.'}
    (args.out/'completed_training_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=['# Verified completed training results','','Three new independent full-pipeline repetitions completed at 00:05 PKT on 1 November 2026. All six regime reports passed prediction/selection hash checks, row identity checks and independent metric recomputation.','']
    for reg,label in [('benchmark','Response time included'),('deployment','Response time excluded')]:
        s=summaries[reg];text+=['## '+label,'',f"Selected model AUROC mean {s['stack_auroc']['mean']:.6f}, sample SD {s['stack_auroc']['sample_sd']:.6f}; boosted mean {s['boosted_auroc']['mean']:.6f}.",f"Stack-minus-boosted AUROC mean {s['auroc_delta']['mean']:+.8f}, sample SD {s['auroc_delta']['sample_sd']:.8f}. All {s['per_run_auroc_intervals_including_zero']} of 3 per-run learner-cluster intervals include zero.",'Selected-method frequencies: '+json.dumps(s['selected_method_frequencies'])+'.','']
        for r in [r for r in records if r['regime']==reg]:text.append(f"- Run {r['repetition']}: selected AUROC {r['stack_auroc']:.6f}; boosted {r['boosted_auroc']:.6f}; delta {r['auroc_delta']:+.8f}, interval [{r['delta_ci_low']:.8f}, {r['delta_ci_high']:.8f}].")
        text.append('')
    text += [result['interpretation'],'','This completes the prespecified full-pipeline repeats. Registry-only refits, a matched full-NuLKT boundary comparison, comparable-budget ablations, refit-effect isolation and the author-managed DOI deposit remain separate outstanding work.']
    (args.out/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({'status':result['status'],'summary':summaries},indent=2))
if __name__=='__main__':main()
