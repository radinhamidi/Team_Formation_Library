import pickle as pkl
import matplotlib.pyplot as plt

k=50

with open('./KL_r_at_k_50.pkl', 'rb') as f:
    KL_r_at_k = pkl.load(f)

with open('./VAE_r_at_k_50.pkl', 'rb') as f:
    VAE_r_at_k = pkl.load(f)

with open('./Baseline_r_at_k_50.pkl', 'rb') as f:
    BL_r_at_k = pkl.load(f)

with open('./T2V_dim2000_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim3000_members_r_at_k = pkl.load(f)

with open('./T2V_dim2000_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim2000_members_r_at_k = pkl.load(f)

with open('./T2V_dim1500_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim1500_members_r_at_k = pkl.load(f)

with open('./T2V_dim1000_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim1000_members_r_at_k = pkl.load(f)

with open('./T2V_dim500_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim500_members_r_at_k = pkl.load(f)

with open('./T2V_dim400_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim400_members_r_at_k = pkl.load(f)

with open('./T2V_dim300_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim300_members_r_at_k = pkl.load(f)

with open('./T2V_dim100_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim100_members_r_at_k = pkl.load(f)

# with open('./T2V_dim300_team_r_at_k_50.pkl', 'rb') as f:
#     T2V_dim300_teams_r_at_k = pkl.load(f)

# with open('./T2V_dim400_team_r_at_k_50.pkl', 'rb') as f:
#     T2V_dim400_teams_r_at_k = pkl.load(f)

# with open('./T2V_dim1000_team_r_at_k_50.pkl', 'rb') as f:
#     T2V_dim1000_teams_r_at_k = pkl.load(f)

with open('T2V_skill_dim500_r_at_k_50.pkl', 'rb') as f:
    T2V_skill_dim500_r_at_k = pkl.load(f)

with open('T2V_skill_KL_dim1000_r_at_k_50.pkl', 'rb') as f:
    T2V_skill_KL_dim1000_r_at_k = pkl.load(f)

with open('T2V_user_VAE_dim1000_member_r_at_k_50.pkl', 'rb') as f:
    T2V_user_VAE_dim1000_r_at_k = pkl.load(f)

with open('T2V_Skill_VAE_Hybrid_r_at_k_50.pkl', 'rb') as f:
    T2V_VAE_hybrid = pkl.load(f)

k_set = range(1,k+1,1)
# Extracting data from json files of records
y_kl = [KL_r_at_k[k] for k in k_set]
y_VAE = [VAE_r_at_k[k] for k in k_set]
y_bl = [BL_r_at_k[k] for k in k_set]
y_t2v_dim3000_members = [T2V_dim3000_members_r_at_k[k] for k in k_set]
y_t2v_dim2000_members = [T2V_dim2000_members_r_at_k[k] for k in k_set]
y_t2v_dim1500_members = [T2V_dim1500_members_r_at_k[k] for k in k_set]
y_t2v_dim1000_members = [T2V_dim1000_members_r_at_k[k] for k in k_set]
y_t2v_dim500_members = [T2V_dim500_members_r_at_k[k] for k in k_set]
y_t2v_dim400_members = [T2V_dim400_members_r_at_k[k] for k in k_set]
y_t2v_dim300_members = [T2V_dim300_members_r_at_k[k] for k in k_set]
y_t2v_dim100_members = [T2V_dim100_members_r_at_k[k] for k in k_set]
# y_t2v_dim300_teams = [T2V_dim300_teams_r_at_k[k] for k in k_set]
# y_t2v_dim400_teams = [T2V_dim400_teams_r_at_k[k] for k in k_set]
# y_t2v_dim1000_teams = [T2V_dim1000_teams_r_at_k[k] for k in k_set]
y_t2v_skill_dim500 = [T2V_skill_dim500_r_at_k[k] for k in k_set]
y_t2v_skill_KL_dim1000 = [T2V_skill_KL_dim1000_r_at_k[k] for k in k_set]
y_t2v_user_VAE_dim1000 = [T2V_user_VAE_dim1000_r_at_k[k] for k in k_set]
y_t2v_skill_VAE_hybrid = [T2V_VAE_hybrid[k] for k in k_set]

plt.figure(0)
plt.xlabel = 'K'
plt.ylabel = 'Recall'
plt.title = 'Recall @ K'
plt.plot(k_set, y_kl, '.-', label='Kullback Leibler Loss Autoencoder')
plt.plot(k_set, y_VAE, '.-', label='Variational Autoencoder')
# plt.plot(k_set, y_t2v_dim3000_members, label='Team2Vec-User - Dim3000 - Member Similarity')
# plt.plot(k_set, y_t2v_dim2000_members, label='Team2Vec-User - Dim2000 - Member Similarity')
# plt.plot(k_set, y_t2v_dim1500_members, label='Team2Vec-User - Dim1500 - Member Similarity')
# plt.plot(k_set, y_t2v_dim1000_members, label='Team2Vec-User - Dim1000 - Member Similarity')
# plt.plot(k_set, y_t2v_dim500_members, label='Team2Vec-User - Dim500 - Member Similarity')
plt.plot(k_set, y_t2v_dim500_members, label='Team2Vec-User - KL')
plt.plot(k_set, y_t2v_user_VAE_dim1000, label='Team2Vec-User - VAE')
# plt.plot(k_set, y_t2v_dim400_members, label='Team2Vec-User - Dim400 - Member Similarity')
# plt.plot(k_set, y_t2v_dim300_members, label='Team2Vec-User - Dim300 - Member Similarity')
# plt.plot(k_set, y_t2v_dim100_members, label='Team2Vec-User - Dim100 - Member Similarity')
# plt.plot(k_set, y_t2v_dim300_teams, label='Team2Vec - Dim300 - Team Similarity')
# plt.plot(k_set, y_t2v_dim400_teams, label='Team2Vec - Dim400 - Team Similarity')
# plt.plot(k_set, y_t2v_dim1000_teams, label='Team2Vec - Dim1000 - Team Similarity')
# plt.plot(k_set, y_t2v_skill_dim500, 'b--', label='Team2Vec-Skill - Dim500')
plt.plot(k_set, y_t2v_skill_KL_dim1000, label='Team2Vec-Skill - KL')
plt.plot(k_set, y_t2v_skill_dim500, 'b--', label='Team2Vec-Skill - VEA')
plt.plot(k_set, y_t2v_skill_VAE_hybrid, label='T2V-Skill-VAE-Hybrid')
plt.plot(k_set, y_bl, label='Baseline Method')
plt.legend(loc='best')
plt.grid()

plt.show()


# print ("{:<8} {:<15} {:<10} {:<10}".format('@K', 'Kullback Leibler',
#                                             'T2V-D300-T',
#                                            'T2V-D500-M', 'T2V-D400-M', 'T2V-D300-M',
#                                            'Baseline'))
# general_dictionary = {'KL': KL_r_at_k, 'T2V-D300-T': T2V_dim300_teams_r_at_k,
#                       'T2V-D300-M': T2V_dim300_members_r_at_k, 'T2V-D400-M': T2V_dim400_members_r_at_k,
#                       'T2V-D500-M': T2V_dim500_members_r_at_k, 'Baseline': BL_r_at_k}

# for k, v in general_dictionary.items():
#     for kk, vv in v.items():
#         print("{:<8} {:<15} {:<10}".format(k, kk, kk))
