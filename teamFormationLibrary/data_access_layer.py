from teamFormationLibrary.dal.embedding import Embedding

class DataAccessLayer:
    def __init__(self, database_name, database_path, embeddings_save_path):
        self.database_name = database_name
        self.database_path = database_path
        self.embeddings_save_path = embeddings_save_path

    def get_database_name(self):
        return self.database_name

    def get_database_path(self):
        return self.database_path

    def embeddings_save_path(self):
        return self.embeddings_save_path

    def generate_embeddings(self):
        if self.embeddings_save_path == 'default':
            self.embedding_model = Embedding(self.database_name, self.database_path)
        else:
            self.embedding_model = Embedding(self.database_name, self.database_path, self.embeddings_save_path)
        self.embedding_model.generate_embeddings()

    # TODO (Step 4):
    def generate_t2v_dataset(self):
        return 0