# -*- coding: utf-8 -*-
"""
Created on Thursday Nov 21 2019

@author: Hossein Fani (sites.google.com/site/hosseinfani/)
"""

import gensim, numpy, pylab, random
import os, getopt, sys, multiprocessing
sys.path.extend(['./../team_formation'])
from Common.tsne import tsne, pca
# teams as documents, members as words
# doc_list = ['u1 u2 u3','u2 u3','u1 u2 u1 u2']
# label_list = ['t1','t2','t3']

class Team2Vec:
    def __init__(self):
        self.teams = []

    def init(self, team_matrix, member_type='user'):#member_type={'user','skill'}
        teams_label = []
        # teams_skils = []
        teams_members = []
        for team in team_matrix:
            teams_label.append(team[0])
            if member_type == 'skill':
                teams_members.append(team[1].col)
            else: #member_type == 'user'
                teams_members.append(team[2].col)

        for index, team in enumerate(teams_members):
            td = gensim.models.doc2vec.TaggedDocument([str(m) for m in team], [
                str(teams_label[index])])  # the f*ing [] is needed to surround the tags!
            self.teams.append(td)
        print('#teams loaded: %s' % len(self.teams))

    def train(self, dimension=300, window=2, dist_mode=1, epochs=100, output='./'):

        self.settings = 'd' + str(dimension) + '_w' + str(window) + '_m' + str(dist_mode)
        print('training settings: %s\n'%self.settings)

        # build the model
        # alpha=0.025
        # min_count=5
        # max_vocab_size=None
        # sample=0
        # seed=1
        # min_alpha=0.0001
        # hs=1
        # negative=0
        # dm_mean=0
        # dm_concat=0
        # dm_tag_count=1
        # docvecs=None
        # docvecs_mapfile=None
        # comment=None
        # trim_rule=None

        self.model = gensim.models.Doc2Vec(dm=dist_mode,#({1,0}, optional) – Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
                                      vector_size=dimension,
                                      window=window,
                                      dbow_words=1, #({1,0}, optional) – If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
                                      min_alpha=0.025,
                                      min_count=0,
                                      workers=multiprocessing.cpu_count())
        self.model.build_vocab(self.teams)

        # start training
        for e in range(epochs):
            if not (e % 10):
                print('iteration {0}'.format(e))
            self.model.train(self.teams, total_examples = self.model.corpus_count, epochs = self.model.epochs)
            self.model.alpha -= 0.002  # decrease the learning rate
            self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay

        if output:
            self.model.save('{}model_{}'.format(output, self.settings))
            self.model.save_word2vec_format('{}members2vec_{}'.format(output, self.settings))
            self.model.docvecs.save_word2vec_format('{}team2vec_{}'.format(output, self.settings))
            print('Model saved for {} under directory {}'.format(self.settings, output))

    def get_member_vec(self, mid):
        return self.model[mid]

    def get_team_vec(self, tid):
        return self.model.docvecs[tid]

    def get_member_similarity(self, m1, m2):
        return self.model.most_similarity(m1, m2)

    def get_team_similarity(self, t1, t2):
        return self.model.docvecs.similarity(t1, t2)

    def get_team_most_similar(self, teamid, topn=10):
        return self.model.docvecs.most_similar(teamid, topn=topn)

    def load_model(self, modelfile):
        self.model = gensim.models.Doc2Vec.load(modelfile)

    def get_member_most_similar_by_vector(self, mvec, topn=10):
        return self.model.wv.similar_by_vector(mvec, topn=topn)

    def get_team_most_similar_by_vector(self, tvec, topn=10):
        return self.model.similar_by_vector(tvec, topn=topn)

    def infer_team_vector(self, members):
        iv = self.model.infer_vector(members)
        return iv, self. model.docvecs.most_similar([iv])

    def plot_model(self, method='pca', memberids=None, teamids=None, output='./'):
        team_vecs = []
        team_labels = []
        member_vecs = []
        member_labels = []

        for member in self.model.wv.vocab.keys():
            if memberids is None or member in memberids:
                member_vecs.append(self.model.wv[member])
                member_labels.append(member)

        for team in self.model.docvecs.doctags.keys():
            if teamids is None or member in teamids:
                team_vecs.append(self.model.docvecs[team])
                team_labels.append(team)

        if method == 'pca':
            members = pca(numpy.array(member_vecs), 2)
            teams = pca(numpy.array(team_vecs), 2)
            all_dw = pca(numpy.array(team_vecs + member_vecs), 2)
        else:
            members = tsne(numpy.array(member_vecs), 2)
            teams = tsne(numpy.array(team_vecs), 2)
            all_dw = tsne(numpy.array(team_vecs + member_vecs), 2)

        # plt.plot(pca.explained_variance_ratio_)
        for index, vec in enumerate(members):
            # print ('%s %s'%(words_label[index], vec))
            pylab.scatter(vec[0], vec[1])
            pylab.annotate(member_labels[index], xy=(vec[0], vec[1]))
        # fig_words_pca = pylab.figure()
        # ax = Axes3D(fig_words_pca)
        # ax.scatter(words_pca[:, 0], words_pca[:, 1], color='r')
        if output:
            pylab.savefig('{}members_{}_{}.png'.format(output, method, self.settings))
        # pylab.show()
        pylab.close()

        for index, vec in enumerate(teams):
            pylab.scatter(vec[0], vec[1])
            pylab.annotate(team_labels[index], xy=(vec[0], vec[1]))

        if output:
            pylab.savefig('{}teams_{}_{}.png'.format(output, method, self.settings))
        pylab.close()

        for index, vec in enumerate(all_dw):
            pylab.scatter(vec[0], vec[1])
            if index < len(member_labels):
                pylab.annotate(member_labels[index], xy=(vec[0], vec[1]))
            else:
                pylab.annotate(team_labels[index - len(member_labels)], xy=(vec[0], vec[1]))
        if output:
            pylab.savefig('{}teams_members_{}_{}.png'.format(output, method, self.settings))
        pylab.close()

if __name__ == "__main__":
    import DataAccessLayer.load_dblp_data as dblp
    if dblp.ae_data_exist():
        team_matrix = dblp.load_ae_dataset()
    else:
        dblp.extract_data(filter_journals=True)
        team_matrix = dblp.load_ae_dataset()

    t2v = Team2Vec()

    help_str = 'team2vec.py [-m] [-s] [-d <dimension=100>] [-w <window=2>] \n-m: distributed memory mode; default=distributed bag of members\n-s: member type = skill; default = user'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hmsd:w:", ["dimension=", "window="])
    except getopt.GetoptError:
        print(help_str)
        sys.exit(2)
    dimension = 100
    window = 2
    dm = 0
    member_type = 'user'
    for opt, arg in opts:
        if opt == '-h':
            print(help_str)
            sys.exit()
        elif opt == '-s':
            member_type = 'skill'
        elif opt == '-m':
            dm = 1
        elif opt in ("-d", "--dimension"):
            dimension = int(arg)
        elif opt in ("-w", "--window"):
            window = int(arg)

    t2v.init(team_matrix, member_type=member_type)
    t2v.train(dimension=dimension, window=window, dist_mode=dm, output='./Output/Team2Vec/', epochs=100)
    t2v.plot_model('pca', output='./Output/Team2Vec/')
    t2v.plot_model('tsne', output='./Output/Team2Vec/')

    #sample running string
    #python3 -u ./Methods/team2vec.py -d 500 -w 2 -m 2>&1 |& tee  ./Output/Team2Vec/log_d500_w2_m1.txt

    #test
    # t2v.init(random.sample(team_matrix, 100))
    # t2v.train(dimension=100, window=2, dist_mode=1, output='./Output/Team2Vec/', epochs=10)
    # t2v.plot_model('pca', output='./Output/Team2Vec/')
    # t2v.plot_model('tsne', output='./Output/Team2Vec/')

