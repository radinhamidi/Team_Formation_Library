class DataAccessLayer:
    def __init__(self, database_name, database_path):
        self.database_name = database_name
        self.databasePath = database_path

    def get_database_name(self):
        return self.database_name

    def get_database_path(self):
        return self.databasePath

    def pre_process_data(self):
        return 1
