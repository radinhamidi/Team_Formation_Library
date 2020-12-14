import os

from teamFormationLibrary.VAE import VAE
from teamFormationLibrary.data_access_layer import DataAccessLayer


def main_team_formation():
    print("---------------------------------------------------------")
    print("This library is a Team Formation tool that uses user database "
          "to predict the best teams to match a specific skill requirement.")
    print("NOTE: the database you provide to this library must be in a one-hot "
          "vector data frame format consisting of the following 3 parts:")
    print("1. ID")
    print("2. Experts")
    print("3. Skills")
    print("---------------------------------------------------------")
    database_name = input("Please enter the name of your database: ")
    database_path = input("Please provide the path of your database: ")
    embeddings_save_path = input("Please enter the path you want to save the "
                                 "embeddings (type 'default' to save it to a "
                                 "default path: ")
    '''
    while not os.path.isdir(embeddings_save_path):
    embeddings_save_path = input("This path does not exist. Please enter a "
                                 "valid path: ")
    '''

    # 1
    DAL = DataAccessLayer(database_name, database_path, embeddings_save_path)
    DAL.generate_embeddings()

    # 2
    '''
    t2v_model = VAE(t2v_model, database_path)
    t2v_model.VAE()
    '''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_team_formation()
