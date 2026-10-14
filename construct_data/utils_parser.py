import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")

    parser.add_argument('--train_path', nargs='?', default='./data/Beidian/data.train',
                        help='Input data path.')
    parser.add_argument('--val_path', nargs='?', default='./data/Beidian/data.val',
                        help='Input data path.')
    parser.add_argument('--test_path', nargs='?', default='./data/Beidian/data.test',
                        help='Input data path.')
    parser.add_argument('--num_user', type=int, default=3773,
                        help='number of users.')
    parser.add_argument('--num_item', type=int, default=4544,
                        help='number of items.')
    parser.add_argument('--dataset', default='Beidian',
                        help='Choose a dataset from {Beidian, Beibei}')
    parser.add_argument('--hyperedge_user_train_path', default='./data/Beidian/MF/hyperedgeUser_train.txt',
                        help='save huperegde user')
    parser.add_argument('--hyperedge_item_train_path', default='./data/Beidian/MF/hyperedgeItem_train.txt',
                        help='save huperegde item')
    parser.add_argument('--top_k', default=5)
    parser.add_argument('--hyperedge_user_topk_train_path', default='./data/Beidian/MF/hyperedgeUser_train_top5.txt')
    parser.add_argument('--hyperedge_item_topk_train_path', default='./data/Beidian/MF/hyperedgeItem_train_top5.txt')
    parser.add_argument('--social_user_train_path', default='./data/Beidian/MF/social_user.txt')
    parser.add_argument('--social_item_train_path', default='./data/Beidian/MF/social_item.txt')
    parser.add_argument('--dataset_save_path', default='./data/Beidian/Beidian.mat')

    return parser.parse_args()


