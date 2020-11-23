from teamFormationLibrary.data_access_layer import DataAccessLayer


def main_team_formation():
    database_name = input("Please enter the name of your database:")
    database_path = input("Please paste the path of your database:")
    DAL = DataAccessLayer(database_name, database_path)
    print(DAL.get_database_name())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_team_formation()
