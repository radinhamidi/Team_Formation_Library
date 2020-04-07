import csv
import numpy as np
import eval.ranking as rk
import ml_metrics as metrics
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval

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
for file_name in file_names:
    method_name, pred_indices, true_indices, calc_user_time, calc_skill_time, k_fold, k_max =\
        dblp_eval.load_output_file(file_name, foldIDsampleID_strata_dict)
    # eval settings
    evaluation_k_set = np.arange(1, k_max + 1, 1)
    fold_set = np.arange(1, k_fold + 1, 1)
    # Initializing metric holders
    Coverage = dblp_eval.init_eval_holder(evaluation_k_set)
    nDCG = dblp_eval.init_eval_holder(evaluation_k_set)
    MAP = dblp_eval.init_eval_holder(evaluation_k_set)
    MRR = dblp_eval.init_eval_holder(evaluation_k_set)
    Quality = dblp_eval.init_eval_holder(evaluation_k_set)
    Hindex_min = dblp_eval.init_eval_holder(evaluation_k_set)
    Hindex_avg = dblp_eval.init_eval_holder(evaluation_k_set)
    Hindex_max = dblp_eval.init_eval_holder(evaluation_k_set)
    Hindex_diff = dblp_eval.init_eval_holder(evaluation_k_set)
    # writing output file
    result_output_name = "../output/eval_results/{}_strata.csv".format(method_name)
    with open(result_output_name, 'w') as file:
        writer = csv.writer(file)

        writer.writerow(['User Quantity Strata Computation Time'])
        for strata in sorted(calc_user_time.keys()):
            writer.writerow(['Strata:', strata, 'Average:',
                             np.mean(calc_user_time[strata]), 'STDev:', np.std(calc_user_time[strata])])

        writer.writerow(['Skill Quantity Strata Computation Time'])
        for strata in sorted(calc_skill_time.keys()):
            writer.writerow(['Strata:', strata, 'Average:',
                             np.mean(calc_skill_time[strata]), 'STDev:', np.std(calc_skill_time[strata])])

        # writer.writerow(['@K',
        #                  'Coverage Mean', 'Coverage STDev',
        #                  'nDCG Mean', 'nDCG STDev',
        #                  'MAP Mean', 'MAP STDev',
        #                  'MRR Mean', 'MRR STDev',
        #                  'Quality Mean', 'Quality STDev',
        #                  'H Index Min Mean', 'H Index Min STDev',
        #                  'H Index Avg Mean', 'H Index Avg STDev',
        #                  'H Index Max Mean', 'H Index Max STDev',
        #                  'H Index Diff Mean', 'H Index Diff STDev'])
        # for i in fold_set:
        #     truth = true_indices[i]
        #     pred = pred_indices[i]
        #     for j in evaluation_k_set:
        #         print('{}, fold {}, @ {}'.format(method_name, i, j))
        #         coverage_overall, _ = dblp_eval.r_at_k(pred, truth, k=j)
        #         Coverage[j].append(coverage_overall)
        #         nDCG[j].append(rk.ndcg_at(pred, truth, k=j))
        #         MAP[j].append(metrics.mapk(truth, pred, k=j))
        #         MRR[j].append(dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score(pred, truth, k=j)))
        #         Quality[j].append(dblp_eval.team_formation_feasibility(pred, truth, user_skill_dict, k=j))
        #         Hindex_min[j].append(dblp_eval.team_formation_feasibility(pred, truth, user_x_dict=user_HIndex,
        #                                                                   mode='hindex', hindex_mode='min', k=j))
        #         Hindex_avg[j].append(dblp_eval.team_formation_feasibility(pred, truth, user_x_dict=user_HIndex,
        #                                                                   mode='hindex', hindex_mode='avg', k=j))
        #         Hindex_max[j].append(dblp_eval.team_formation_feasibility(pred, truth, user_x_dict=user_HIndex,
        #                                                                   mode='hindex', hindex_mode='max', k=j))
        #         Hindex_diff[j].append(dblp_eval.team_formation_feasibility(pred, truth, user_x_dict=user_HIndex,
        #                                                                    mode='hindex', hindex_mode='diff', k=j))
        #
        # for j in evaluation_k_set:
        #     Coverage_mean = np.mean(Coverage[j])
        #     Coverage_std = np.std(Coverage[j])
        #
        #     nDCG_mean = np.mean(nDCG[j])
        #     nDCG_std = np.std(nDCG[j])
        #
        #     MAP_mean = np.mean(MAP[j])
        #     MAP_std = np.std(MAP[j])
        #
        #     MRR_mean = np.mean(MRR[j])
        #     MRR_std = np.std(MRR[j])
        #
        #     Quality_mean = np.mean(Quality[j])
        #     Quality_std = np.std(Quality[j])
        #
        #     Hindex_min_mean = np.mean(Hindex_min[j])
        #     Hindex_min_std = np.std(Hindex_min[j])
        #
        #     Hindex_avg_mean = np.mean(Hindex_avg[j])
        #     Hindex_avg_std = np.std(Hindex_avg[j])
        #
        #     Hindex_max_mean = np.mean(Hindex_max[j])
        #     Hindex_max_std = np.std(Hindex_max[j])
        #
        #     Hindex_diff_mean = np.mean(Hindex_diff[j])
        #     Hindex_diff_std = np.std(Hindex_diff[j])
        #
        #     writer.writerow([j,
        #                      Coverage_mean, Coverage_std,
        #                      nDCG_mean, nDCG_std,
        #                      MAP_mean, MAP_std,
        #                      MRR_mean, MRR_std,
        #                      Quality_mean, Quality_std,
        #                      Hindex_min_mean, Hindex_min_std,
        #                      Hindex_avg_mean, Hindex_avg_std,
        #                      Hindex_max_mean, Hindex_max_std,
        #                      Hindex_diff_mean, Hindex_diff_std])
        #
        # file.close()
