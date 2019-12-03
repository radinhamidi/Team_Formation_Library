import pickle as pkl
import matplotlib.pyplot as plt

with open('./KL_r_at_k_50.pkl', 'rb') as f:
    KL_r_at_k = pkl.load(f)

with open('./Baseline_r_at_k_50.pkl', 'rb') as f:
    BL_r_at_k = pkl.load(f)

with open('./T2V_dim500_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim500_members_r_at_k = pkl.load(f)

with open('./T2V_dim400_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim400_members_r_at_k = pkl.load(f)

with open('./T2V_dim300_member_r_at_k_50.pkl', 'rb') as f:
    T2V_dim300_members_r_at_k = pkl.load(f)


k_set = range(1,51,1)
# Extracting data from json files of records
y_kl = [KL_r_at_k[k] for k in k_set]
y_bl = [BL_r_at_k[k] for k in k_set]
y_t2v_dim500_members = [T2V_dim500_members_r_at_k[k] for k in k_set]
y_t2v_dim400_members = [T2V_dim400_members_r_at_k[k] for k in k_set]
y_t2v_dim300_members = [T2V_dim300_members_r_at_k[k] for k in k_set]

plt.figure(0)
plt.xlabel = 'K'
plt.ylabel = 'Recall'
plt.title = 'Recall @ K'
plt.plot(k_set, y_kl, label='Kullback Leibler')
plt.plot(k_set, y_t2v_dim500_members, label='Team2Vec - Dim500 - Member Similarity')
plt.plot(k_set, y_t2v_dim400_members, label='Team2Vec - Dim400 - Member Similarity')
plt.plot(k_set, y_t2v_dim300_members, label='Team2Vec - Dim300 - Member Similarity')
plt.plot(k_set, y_bl, label='Baseline Method')
plt.legend(loc='best')
plt.grid()

plt.show()