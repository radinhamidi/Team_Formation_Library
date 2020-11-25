from teamFormationLibrary.data_access_layer import DataAccessLayer


def main_team_formation():
    print("---------------------------------------------------------")
    print("This library is a Team Formation tool that uses user database "
          "to predict the best teams to match a specific skill requirement."
          )
    print("NOTE: the database you provide to this library must be in a one-hot "
          "vector data frame format consisting of the following 3 parts:"
          )
    print("1. ID")
    print("2. Experts")
    print("3. Skills")
    print("---------------------------------------------------------")
    database_name = input("Please enter the name of your database: ")
    database_path = input("Please provide the path of your database: ")
    DAL = DataAccessLayer(database_name, database_path)
    DAL.pre_process_data()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_team_formation()
