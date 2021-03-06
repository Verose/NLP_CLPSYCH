import glob
import json
import optparse
import os
import time
from tqdm import tqdm
from multiprocessing import Manager
from multiprocessing.pool import Pool

from annoy import AnnoyIndex
from sklearn.preprocessing import StandardScaler

from common.utils import DATA_DIR


def load_embeddings(json_files):
    index = 0
    embeddings = []

    for i, file in enumerate(json_files):
        user_id = os.path.basename(file).split('.')[0]

        with open(file, encoding='utf-8') as f:
            data = json.load(f)
            posts_embeddings = data['embeddings']

            for post_ind in range(len(posts_embeddings)):
                emb_ind_to_user_n_post_ind[index] = (user_id, post_ind)
                index += 1
            embeddings.extend(posts_embeddings)
        if i % 150 == 0 and i > 0:
            print("Finished loading {} users".format(i))
    print("*******Finished loading all {} of the vectors*******".format(len(embeddings)))

    return embeddings


def create_filtered_jsons(filtered_embeddings, embeds_dir, out_dir):
    print("*******Creating filtered jsons*******")
    user_post_inds = []
    user = ''
    prev_user = ''
    is_first = True
    print("Finished writing users: ", end='')

    for emb_ind in sorted(filtered_embeddings):
        user, point_ind = emb_ind_to_user_n_post_ind[str(emb_ind)]
        if is_first:
            prev_user = user
            is_first = False

        if user == prev_user:
            user_post_inds.append(point_ind)
        else:
            # create jsons for previous user
            write_filtered_jsons(prev_user, user_post_inds, embeds_dir, out_dir)
            prev_user = user
            user_post_inds = [point_ind]
    write_filtered_jsons(user, user_post_inds, embeds_dir, out_dir)
    print('\n')


def write_filtered_jsons(user, inds, embeds_dir, out_path):
    with open(os.path.join(embeds_dir, '{}.json').format(user),
              encoding='utf-8') as f:
        user_data = {}
        data = json.load(f)
        user_data['label'] = data['label']
        user_data['tokens'] = [data['tokens'][i] for i in inds]
        user_data['posTags'] = [data['posTags'][i] for i in inds]
        user_data['embeddings'] = [data['embeddings'][i] for i in inds]

        with open(os.path.join(out_path, '{}.json'.format(user)), 'w') as out_file:
            json.dump(user_data, out_file)
    print("{}, ".format(user), end='')


def init(*args):
    """ store the dataset for later use """
    global cores_list
    global cores_neighbors
    global run_times
    cores_list = args[0]
    cores_neighbors = args[1]
    run_times = args[2]


run_times = []
cores_list = []
cores_neighbors = {}


def check_is_core_point(i, min_samples, eps, vector_dim, annoy_save_path):
    global cores_list
    global run_times
    annoy_load = AnnoyIndex(vector_dim, 'euclidean')
    annoy_load.load(annoy_save_path)

    ts = time.time()
    neighbors, distances = annoy_load.get_nns_by_item(i, min_samples, search_k=-1, include_distances=True)
    te = time.time()
    run_times.append(int((te - ts) * 1000))
    if sum([1 for dist in distances if dist < eps]) == min_samples:
        cores_list.append(i)
        for neighbor in neighbors:
            cores_neighbors[neighbor] = True


if __name__ == "__main__":
    print("*******Starting to run!*******")
    parser = optparse.OptionParser()
    parser.add_option('--eps', action="store", type=int)
    parser.add_option('--min_samples', action="store", type=int)
    parser.add_option('--trees', action="store", type=int, default=10000)
    parser.add_option('--output', action="store", type=str, default="")
    parser.add_option('--n_processes', action="store", type=int, default=1)
    parser.add_option('--dataset', choices=['rsdd', 'smhd', 'tssd'], default='rsdd', action="store")
    options, _ = parser.parse_args()

    eps = options.eps
    min_samples = options.min_samples
    num_trees = options.trees
    print("Using eps={}, min_samples={}".format(eps, min_samples))

    embeddings_dir = os.path.join('..', DATA_DIR, 'pos_tags_{}_embeds'.format(options.dataset))
    annoy_output_dir = os.path.join('..', DATA_DIR, 'annoy_filtered_{}'.format(options.dataset), options.output)
    if not os.path.isdir(annoy_output_dir):
        os.mkdir(annoy_output_dir)

    emb_ind_to_user_n_post_ind = {}
    print("*******Creating Annoy (if needed)*******")
    annoy_save_path = os.path.join(annoy_output_dir, 'annoy_{}.ann'.format(options.dataset))
    mapping_save_path = os.path.join(annoy_output_dir, 'mapping_{}.json'.format(options.dataset))
    vector_dim = 2148

    if not os.path.isfile(annoy_save_path):
        print("*******Loading embeddings*******")
        json_pattern = os.path.join(embeddings_dir, '*.json')
        json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

        all_embeddings = load_embeddings(json_files)

        print("*******Standardizing the data*******")
        X = StandardScaler().fit_transform(all_embeddings)
        # X = X.astype(np.float32)

        print("*******Saving Annoy data*******")
        annoy = AnnoyIndex(vector_dim, 'euclidean')
        for i, sample in enumerate(X):
            annoy.add_item(i, sample)
        annoy.build(num_trees)
        annoy.save(annoy_save_path)

        with open(mapping_save_path, 'w') as out_file:
            json.dump(emb_ind_to_user_n_post_ind, out_file)
    else:
        annoy = AnnoyIndex(vector_dim, 'euclidean')
        annoy.load(annoy_save_path)

    print("*******Finding core samples*******")
    # start algorithm
    # for each vector: mark it as 'core' if it has at least 'min_samples' neighbors within radius 'eps'
    num_samples = annoy.get_n_items()

    manager = Manager()
    cores_list = manager.list()
    cores_neighbors = manager.dict()
    run_times = manager.list()

    def update(*args):
        pbar.update()

    pool = Pool(processes=options.n_processes, initializer=init, initargs=(cores_list, cores_neighbors, run_times))
    pbar = tqdm(range(num_samples), total=num_samples, leave=False, desc='Searching Core Samples')
    for i in range(pbar.total):
        pool.apply_async(check_is_core_point, args=(i, min_samples, eps, vector_dim, annoy_save_path), callback=update)
    pool.close()
    pool.join()
    pbar.close()
    print("Average search time: {}ms over {} samples".format(float(sum(run_times) / num_samples), num_samples))
    print("Found {} core samples".format(len(cores_list)))

    print("*******Finding neighbors of core samples*******")
    # for each non-core vector check if it has a 'core' neighbor within radius 'eps'
    for neighbor in cores_neighbors.keys():
        if neighbor not in cores_list:
            cores_list.append(neighbor)
    print("Found a total of {} core samples".format(len(cores_list)))
    print("*******Finished fitting the data*******")

    print('Estimated number of noise points: %d' % len(set(range(num_samples)) - set(cores_list)))

    if not emb_ind_to_user_n_post_ind:
        with open(mapping_save_path) as in_file:
            emb_ind_to_user_n_post_ind = json.load(in_file)

    create_filtered_jsons(cores_list, embeddings_dir, annoy_output_dir)
