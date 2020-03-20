import csv
import numpy as np
import eval.ranking as rk
import ml_metrics as metrics
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval

user_HIndex = dblp.get_user_HIndex()
user_skill_dict = dblp.get_user_skill_dict(dblp.load_preprocessed_dataset())

OKLO = '../output/predictions/O_KL_O_2020_03_18-09_25_38.csv'
OKLU = '../output/predictions/O_KL_U_2020_03_18-06_11_57.csv'
OVAEO = '../output/predictions/O_VAE_O_2020_03_18-06_53_07.csv'
OVAEU = '../output/predictions/O_VAE_U_2020_03_18-06_05_58.csv'
SKLO = '../output/predictions/S_KL_O_2020_03_17-22_48_21.csv'
SKLU = '../output/predictions/S_KL_U_2020_03_18-06_39_48.csv'
SVAEO = '../output/predictions/S_VAE_O_2020_03_17-12_53_29.csv'
SVAEU = '../output/predictions/S_VAE_U_2020_03_18-06_27_10.csv'
Sapienza = '../output/predictions/Sapienza_2020_03_17-11_45_52.csv'
SVDpp = '../output/predictions/SVDpp_2020_03_18-15_16_00.csv'

file_names = [OKLO, OKLU, OVAEO, OVAEU, SKLO, SKLU, SVAEO, SVAEU, Sapienza, SVDpp]
for file_name in file_names:
    method_name, pred_indices, true_indices, calc_time, k_fold, k_max = dblp_eval.load_output_file(file_name)
    # eval settings
    evaluation_k_set = np.arange(1, k_max + 1, 1)
    fold_set = np.arange(1, k_fold + 1, 1)
    # Initializing metric holders
    Coverage = dblp_eval.init_eval_holder(evaluation_k_set)
    nDCG = dblp_eval.init_eval_holder(evaluation_k_set)
    MAP = dblp_eval.init_eval_holder(evaluation_k_set)
    MRR = dblp_eval.init_eval_holder(evaluation_k_set)
    Quality = dblp_eval.init_eval_holder(evaluation_k_set)
    # writing output file
    result_output_name = "../output/eval_results/{}.csv".format(method_name)
    with open(result_output_name, 'w') as file:
        writer = csv.writer(file)
        for strata in sorted(calc_time.keys()):
            writer.writerow(['Strata:', strata, 'Average:', np.mean(calc_time[strata]), 'STDev:', np.std(calc_time[strata])])
        writer.writerow(['@K', 'Coverage Mean', 'Coverage STDev',
                         'nDCG Mean', 'nDCG STDev',
                         'MAP Mean', 'MAP STDev',
                         'MRR Mean', 'MRR STDev',
                         'Quality Mean', 'Quality STDev',
                         'H Index'])
        for i in fold_set:
            truth = true_indices[i]
            pred = pred_indices[i]
            for j in evaluation_k_set:
                print('{}, fold {}, @ {}'.format(method_name, i, j))
                coverage_overall, _ = dblp_eval.r_at_k(pred, truth, k=j)
                Coverage[j].append(coverage_overall)
                nDCG[j].append(rk.ndcg_at(pred, truth, k=j))
                MAP[j].append(metrics.mapk(truth, pred, k=j))
                MRR[j].append(dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score(pred, truth, k=j)))
                Quality[j].append(dblp_eval.team_formation_feasiblity(pred, truth, user_skill_dict, k=j))

        for j in evaluation_k_set:
            Coverage_mean = np.mean(Coverage[j])
            Coverage_std = np.std(Coverage[j])
            nDCG_mean = np.mean(nDCG[j])
            nDCG_std = np.std(nDCG[j])
            MAP_mean = np.mean(MAP[j])
            MAP_std = np.std(MAP[j])
            MRR_mean = np.mean(MRR[j])
            MRR_std = np.std(MRR[j])
            Quality_mean = np.mean(Quality[j])
            Quality_std = np.std(Quality[j])

            writer.writerow([j, Coverage_mean, Coverage_std, nDCG_mean, nDCG_std, MAP_mean, MAP_std, MRR_mean, MRR_std, Quality_mean, Quality_std])

        file.close()


