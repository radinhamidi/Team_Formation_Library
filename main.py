import os


def main():
    # Running modules

    os.system('python ./ml/nn_KL.py')
    os.system('python ./ml/nn_VAE.py')
    os.system('python ./ml/nn_t2v_user_KL.py')
    os.system('python ./ml/nn_t2v_user_VAE.py')
    os.system('python ./ml/nn_t2v_full_KL.py')
    os.system('python ./ml/nn_t2v_full_VAE.py')
    os.system('python ./ml/nn_t2v_skill_KL.py')
    os.system('python ./ml/nn_t2v_skill_VAE.py')
    os.system('python ./ml/nn_Hybrid.py')

    # Evaluation


if __name__ == '__main__':
    main()
