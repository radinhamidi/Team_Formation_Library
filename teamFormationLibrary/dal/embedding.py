import gensim, numpy, pylab, random, pickle
import os, getopt, sys, multiprocessing

import teamFormationLibrary.dal.load_dblp_data as dblp


class Embedding:
    def __init__(self, database_name, database_path, embeddings_save_path='output/Models/T2V/'):
        self.database_name = database_name
        self.databasePath = database_path
        self.embeddings_save_path = embeddings_save_path
        self.teams = []
        self.member_type = ''

    def get_database_name(self):
        return self.database_name

    def get_database_path(self):
        return self.databasePath

    def embeddings_save_path(self):
        return self.embeddings_save_path

    def init(self, team_matrix, member_type='user'):  # member_type={'user','skill'}
        self.member_type = member_type
        teams_label = []
        # teams_skils = []
        teams_members = []
        for team in team_matrix:
            teams_label.append(team[0])
            if member_type.lower() == 'skill':
                teams_members.append(team[1].col)
            else:  # member_type == 'user'
                teams_members.append(team[2].col)

        for index, team in enumerate(teams_members):
            td = gensim.models.doc2vec.TaggedDocument([str(m) for m in team], [
                str(teams_label[index])])  # the [] is needed to surround the tags!
            self.teams.append(td)
        print('#teams loaded: {}; member type = {}'.format(len(self.teams), member_type))

    def train(self, dimension=300, window=2, dist_mode=1, epochs=100, output=embeddings_save_path):

        self.settings = 'd' + str(dimension) + '_w' + str(window) + '_m' + str(dist_mode) + '_t' + str(self.member_type.capitalize())
        print('training settings: %s\n' % self.settings)

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

        self.model = gensim.models.Doc2Vec(dm=dist_mode,
                                           # ({1,0}, optional) – Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
                                           vector_size=dimension,
                                           window=window,
                                           dbow_words=1,
                                           # ({1,0}, optional) – If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
                                           min_alpha=0.025,
                                           min_count=0,
                                           workers=multiprocessing.cpu_count())
        self.model.build_vocab(self.teams)

        # start training
        for e in range(epochs):
            if not (e % 10):
                print('iteration {0}'.format(e))
            self.model.train(self.teams, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            self.model.alpha -= 0.002  # decrease the learning rate
            self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay

        if output:
            with open('{}teams_{}'.format(output, self.settings), 'wb') as f:
                pickle.dump(self.teams, f)
            self.model.save('{}model_{}'.format(output, self.settings))
            self.model.save_word2vec_format('{}members2vec_{}'.format(output, self.settings))
            self.model.docvecs.save_word2vec_format('{}team2vec_{}'.format(output, self.settings))
            print('Model saved for {} under directory {}'.format(self.settings, output))

    def get_teams(self):
        return self.model.docvecs.doctags

    def get_members(self):
        return self.model.wv.vocab

    def get_team_members(self, tid):
        return [int(m) for t in self.teams if str(tid) in t.tags for m in t.words]

    def get_member_vec(self, mid):
        return self.model[str(mid)]

    def get_team_vec(self, tid):
        return self.model.docvecs[str(tid)]

    def get_member_similarity(self, m1, m2):
        return self.model.wv.similarity(str(m1), str(m2))

    def get_team_similarity(self, t1, t2):
        return self.model.docvecs.similarity(str(t1), str(t2))

    def get_team_most_similar(self, tid, topn=10):
        return self.model.docvecs.most_similar(str(tid), topn=topn)

    def load_model(self, modelfile, includeTeams=False):
        # ModuleNotFoundError: No module named 'numpy.random._pickle': numpy version conflict when saving and loading
        self.model = gensim.models.Doc2Vec.load(modelfile)
        if includeTeams:
            with open(modelfile.replace('model', 'teams'), 'rb') as f:
                self.teams = pickle.load(f)

    def get_member_most_similar_by_vector(self, mvec, topn=10):
        similar_list = self.model.wv.similar_by_vector(mvec, topn=topn)  # is it sorted?
        similar_list.sort(key=lambda x: x[1], reverse=True)  # now it is sorted :)
        return similar_list

    def get_team_most_similar_by_vector(self, tvec, topn=10):
        similar_list = self.model.similar_by_vector(tvec, topn=topn)  # is it sorted?
        similar_list.sort(key=lambda x: x[1], reverse=True)  # now it is sorted :)
        return similar_list

    def infer_team_vector(self, members):
        iv = self.model.infer_vector(members)
        return iv, self.model.docvecs.most_similar([iv])

    def generate_embeddings(self):
        min_skill_size = 0
        min_member_size = 0

        if dblp.preprocessed_dataset_exist(self.databasePath):
            team_matrix = dblp.load_preprocessed_dataset(self.databasePath) #added this
            #dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)

            help_str = 'team2vec.py [-m] [-s] [-d <dimension=100>] [-e <epochs=100>] [-w <window=2>] \n-m: distributed memory mode; default=distributed bag of members\n-s: member type = skill; default = user'
            try:
                opts, args = getopt.getopt(sys.argv[1:], "hmsd:w:", ["dimension=", "window="])
            except getopt.GetoptError:
                print(help_str)
                sys.exit(2)
            dimension = 100
            epochs = 100
            window = 2
            dm = 0
            member_type = 'skill' #added this
            #member_type = 'user'
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
                elif opt in ("-e", "--epochs"):
                    epochs = int(arg)
                elif opt in ("-w", "--window"):
                    window = int(arg)

            self.init(team_matrix, member_type=member_type)
            self.train(dimension=dimension, window=window, dist_mode=dm, output=self.embeddings_save_path, epochs=epochs)

            # sample running string
            # python3 -u ./ml/team2vec.py -d 500 -w 2 -m 2>&1 |& tee  ./output/Team2Vec/log_d500_w2_m1.txt

            # test
            # self.init(random.sample(team_matrix, 100))
            # self.train(dimension=100, window=2, dist_mode=1, output='./output/Team2Vec/', epochs=10)

        else:
            print("The preprocessed database provided does not exist!")
