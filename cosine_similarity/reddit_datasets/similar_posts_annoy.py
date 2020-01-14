import glob
import json
import optparse
import os
import time
from tqdm import tqdm
from operator import itemgetter

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
        if i % 10 == 0 and i > 0:
            print("Finished loading {} users".format(i))
            break
    print("*******Finished loading all {} of the vectors*******".format(len(embeddings)))

    return embeddings


def create_filtered_jsons(filtered_embeddings, embeds_dir, out_dir):
    print("*******Creating filtered jsons*******")
    user_post_inds = []
    user = ''
    prev_user = ''
    is_first = True
    print("Finished writing users: ", end='')

    for emb_ind in filtered_embeddings:
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
        slicer = itemgetter(*inds)
        user_data['tokens'] = slicer(data['tokens'])
        user_data['posTags'] = slicer(data['posTags'])
        user_data['embeddings'] = slicer(data['embeddings'])

        with open(os.path.join(out_path, '{}.json'.format(user)), 'w') as out_file:
            json.dump(user_data, out_file)
    print("{}, ".format(user), end='')


if __name__ == "__main__":
    print("*******Starting to run!*******")
    parser = optparse.OptionParser()
    parser.add_option('--eps', action="store", type=int)
    parser.add_option('--min_samples', action="store", type=int)
    parser.add_option('--trees', action="store", type=int, default=10000)
    parser.add_option('--output', action="store", type=str, default="")
    parser.add_option('--dataset', choices=['rsdd', 'smhd'], default='rsdd', action="store")
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
    mapping_save_path = os.path.join(annoy_output_dir, 'mappong_{}.json'.format(options.dataset))
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
        annoy_save = AnnoyIndex(vector_dim, 'euclidean')
        for i, sample in enumerate(X):
            annoy_save.add_item(i, sample)
        annoy_save.build(num_trees)
        annoy_save.save(annoy_save_path)

        with open(mapping_save_path, 'w') as out_file:
            json.dump(emb_ind_to_user_n_post_ind, out_file)

    annoy_load = AnnoyIndex(vector_dim, 'euclidean')
    annoy_load.load(annoy_save_path)
    if not emb_ind_to_user_n_post_ind:
        with open(mapping_save_path) as in_file:
            emb_ind_to_user_n_post_ind = json.load(in_file)

    print("*******Finding core samples*******")
    run_times = []
    # start algorithm
    # for each vector: mark it as 'core' if it has at least 'min_samples' neighbors within radius 'eps'
    cores = []
    num_samples = annoy_load.get_n_items()

    for i in tqdm(range(num_samples), total=num_samples, leave=False, desc='Searching Core Samples'):
        ts = time.time()
        neighbors, distances = annoy_load.get_nns_by_item(i, min_samples, search_k=-1, include_distances=True)
        te = time.time()
        run_times.append(int((te - ts) * 1000))
        if sum([1 for dist in distances if dist < eps]) == min_samples:
            cores.append(i)

    print("Average search time: {}ms over {} samples".format(float(sum(run_times) / num_samples), num_samples))

    print("*******Finding neighbors of core samples*******")
    # for each non-core vector check if it has a 'core' neighbor within radius 'eps'
    fixed_cores = []
    for non_core in set(range(num_samples)) - set(cores):
        for core in cores:
            if annoy_load.get_distance(non_core, core) < eps:
                fixed_cores.append(non_core) if non_core not in fixed_cores else fixed_cores
    cores.extend(fixed_cores)

    print("*******Finished fitting the data*******")

    print('Estimated number of noise points: %d' % len(set(range(num_samples)) - set(cores)))

    create_filtered_jsons(cores, embeddings_dir, annoy_output_dir)
