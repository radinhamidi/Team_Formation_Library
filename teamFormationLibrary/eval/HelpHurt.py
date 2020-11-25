import csv
import numpy as np
import eval.ranking as rk
import ml_metrics as metrics
import matplotlib.pyplot as plt
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval

filter_zero = True
at_k_mx = 10
at_k_set = range(1,at_k_mx+1,1)

user_HIndex = dblp.get_user_HIndex()
user_skill_dict = dblp.get_user_skill_dict(dblp.load_preprocessed_dataset())
foldIDsampleID_strata_dict = dblp.get_foldIDsampleID_stata_dict(data=dblp.load_preprocessed_dataset(),
                                                                train_test_indices=dblp.load_train_test_indices(),
                                                                kfold=10)

OKLO = '../output/predictions/O_KL_O_output.csv'
OKLU = '../output/predictions/O_KL_U_output.csv'
OVAEO = '../output/predictions/O_VAE_O_output.csv'
OVAEU = '../output/predictions/O_VAE_U_output.csv'
SKLO = '../output/predictions/S_KL_O_output.csv'
SKLU = '../output/predictions/S_KL_U_output.csv'
SVAEO = '../output/predictions/S_VAE_O_output.csv'
SVAEU = '../output/predictions/S_VAE_U_output.csv'
Sapienza = '../output/predictions/Sapienza_output.csv'
SVDpp = '../output/predictions/SVDpp_output.csv'
RRN = '../output/predictions/RRN_output.csv'
BL2009 = '../output/predictions/BL2009_output.csv'
BL2017 = '../output/predictions/BL2017_output.csv'

file_names = [OKLO, OKLU, OVAEO, OVAEU, SKLO, SKLU, SVAEO, SVAEU, Sapienza, SVDpp, RRN, BL2009, BL2017]
# Asking models
models = input('Enter pair to compare separated by space:\n' + '\n'.join(['{}. {}'.format(i+1, f.split('/')[-1].replace('_output.csv', '')) for i, f in enumerate(file_names)]) + '\n')
model1 = int(models.split()[0]) - 1
model2 = int(models.split()[1]) - 1
file_name1 = file_names[model1]
file_name2 =file_names[model2]

# loading models outputs
method_name1, pred_indices1, true_indices1, _, _, k_fold1, _ = dblp_eval.load_output_file(file_name1, foldIDsampleID_strata_dict)
method_name2, pred_indices2, true_indices2, _, _, _, _ = dblp_eval.load_output_file(file_name2, foldIDsampleID_strata_dict)

# eval settings
fold_set = np.arange(1, k_fold1 + 1, 1)

# Initializing metric holders
holder_ndcg = dblp_eval.init_eval_holder(at_k_set)
holder_map = dblp_eval.init_eval_holder(at_k_set)
holder_mrr = dblp_eval.init_eval_holder(at_k_set)

# calculating the diff
for k in at_k_set:
    for i in fold_set:
        truth1 = true_indices1[i]
        pred1 = pred_indices1[i]

        truth2 = true_indices2[i]
        pred2 = pred_indices2[i]

        print('{} & {}, fold {}, @ {}'.format(method_name1, method_name2, i, k))
        # if metric.lower() is 'coverage':
        #     coverage_overall, _ = dblp_eval.r_at_k(pred, truth, k=k)
        #     holder[k].append(coverage_overall)

        holder_ndcg[k].extend([rk.ndcg_at([p1], [t1], k=k) - rk.ndcg_at([p2], [t2], k=k) for
                          p1, t1, p2, t2 in zip(pred1, truth1, pred2, truth2)])
        holder_map[k].extend([metrics.mapk([p1], [t1], k=k) - metrics.mapk([p2], [t2], k=k)
                          for p1, t1, p2, t2 in zip(pred1, truth1, pred2, truth2)])
        holder_mrr[k].extend([dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score([p1], [t1], k=k)) -
                          dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score([p2], [t2], k=k))
                          for p1, t1, p2, t2 in zip(pred1, truth1, pred2, truth2)])

NDCG = np.mean(list(holder_ndcg.values()), axis=0)
MAP = np.mean(list(holder_map.values()), axis=0)
MRR = np.mean(list(holder_mrr.values()), axis=0)
# writing output file
result_output_name = "../output/eval_results/HelpHurt_{}_{}.csv".format(method_name1, method_name2)
with open(result_output_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['NDCG', 'MAP', 'MRR'])
    for j in range(len(NDCG)):
        writer.writerow([NDCG[j], MAP[j], MRR[j]])
    file.close()
print('File saved. Under the name: ', result_output_name)

# Preview test code
plt.title = 'Help Hurt Plot'
plt.xlabel = 'Sample #'
diff = np.sort(NDCG)[::-1]
if filter_zero: diff = diff[diff.nonzero()]
for i, j in enumerate(diff):
    plt.bar(i, j, color='b', width=1)
plt.plot(range(len(diff)), np.zeros(len(diff)), color='peachpuff')
plt.xlim(0, len(diff))
plt.ylim(-1.3*max(diff), 1.3*max(diff))
plt.grid()
plt.show()
