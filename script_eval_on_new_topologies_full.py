"""
Script con để đánh giá một instance (TM) độc lập trên topology mới

Được gọi bởi eval_on_new_topologies_full.py

Cách chạy:
    python3 script_eval_on_new_topologies_full.py -t 0 -m 2 -g Geant2001 -o ../results/ -d "Enero_3top_15_B_NEW" -f ../dataset/Geant2001
"""

import numpy as np
import gym
import os
import gc
import gym_graph
import random
import argparse
import tensorflow as tf
import actorPPOmiddR as actor
from keras import backend as K
import pickle
import sys

sys.setrecursionlimit(2000)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ENV_NAME = 'GraphEnv-v16'
SEED = 9
EPISODE_LENGTH = 100
NUM_ACTIONS = 100

percentage_demands = 15
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(1)

hparams = {
    'l2': 0.0001,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.0002,
    'T': 5,
}

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class DRLAgent:
    def __init__(self, checkpoint_dir, model_id):
        self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        self.actor.build()

        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], beta_1=0.9, epsilon=1e-05)
        checkpoint_actor = tf.train.Checkpoint(model=self.actor, optimizer=optimizer)
        try:
            checkpoint_actor.restore(checkpoint_dir + f"/ckpt_ACT-{model_id}")
            print(f" Đã nạp model từ: {checkpoint_dir}/ckpt_ACT-{model_id}")
        except Exception as e:
            print(f"  Không thể nạp model: {e}")

    def pred_action_distrib_sp(self, env, source, destination):
        """Dự đoán phân phối action."""
        list_k_features = list()
        middlePointList = env.src_dst_k_middlepoints[str(source) + ':' + str(destination)]

        for midpoint in middlePointList:
            env.mark_action_sp(source, midpoint, source, destination)
            if midpoint != destination:
                env.mark_action_sp(midpoint, destination, source, destination)

            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)
            env.edge_state[:, 2] = 0

        vs = list_k_features

        try:
            graph_ids = [tf.fill([tf.shape(vs[i]['link_state'])[0]], i) for i in range(len(vs))]
            first_offset = old_cummax(vs, lambda v: v['first'])
            second_offset = old_cummax(vs, lambda v: v['second'])

            tensor = {
                'graph_id': tf.concat([v for v in graph_ids], axis=0),
                'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
                'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
                'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
                'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }

            r = self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'],
                          tensor['second'], tensor['num_edges'], training=False)
            qvalues = tf.reshape(r, (1, len(r)))
            softmax_probs = tf.nn.softmax(qvalues)
            return softmax_probs.numpy()[0], tensor
        except Exception as e:
            print(f"  Lỗi khi dự đoán: {e}")
            return np.ones(len(middlePointList)) / len(middlePointList), None

    def get_graph_features(self, env, source, destination):
        """Lấy đặc trưng của đồ thị."""
        bw_allocated_feature = env.edge_state[:, 2]
        utilization_feature = env.edge_state[:, 0]

        sample = {
            'num_edges': env.numEdges,
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'bw_allocated': tf.convert_to_tensor(bw_allocated_feature, dtype=tf.float32),
            'utilization': tf.convert_to_tensor(
                np.divide(utilization_feature, env.edge_state[:, 1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(
            sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(
            sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(
            sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat(
            [sample['utilization'], sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3]])
        link_state = tf.pad(hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {
            'link_state': link_state,
            'first': sample['first'][0:sample['length']],
            'second': sample['second'][0:sample['length']],
            'num_edges': sample['num_edges']
        }

        return inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval một instance trên topology mới')

    parser.add_argument('-t', help='TM ID', type=int, required=True, nargs='+')
    parser.add_argument('-m', help='Model ID', type=int, required=True, nargs='+')
    parser.add_argument('-g', help='Graph topology name', type=str, required=True, nargs='+')
    parser.add_argument('-o', help='Output folder', type=str, required=True, nargs='+')
    parser.add_argument('-d', help='Differentiation string', type=str, required=True, nargs='+')
    parser.add_argument('-f', help='Dataset folder', type=str, required=True, nargs='+')

    args = parser.parse_args()

    tm_id = args.t[0]
    model_id = args.m[0]
    topology_name = args.g[0]
    output_folder = args.o[0]
    diff_str = args.d[0]
    dataset_folder = args.f[0]

    print(f"\n{'='*60}")
    print(f" Đang đánh giá {topology_name} - TM {tm_id}")
    print(f"{'='*60}")

    try:
        env = gym.make(ENV_NAME)
        env.seed(SEED)
        env.generate_environment(dataset_folder, topology_name, EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
        env.top_K_critical_demands = True
        print(f" Đã nạp environment: {topology_name}")
    except Exception as e:
        print(f" Lỗi khi nạp environment: {e}")
        sys.exit(1)

    checkpoint_dir = "./models/" + diff_str
    try:
        agent = DRLAgent(checkpoint_dir, model_id)
    except Exception as e:
        print(f" Lỗi khi nạp agent: {e}")
        sys.exit(1)

    try:
        demand, source, destination = env.reset(tm_id)
        done = False
        total_reward = 0
        step_count = 0
        error_links = 0
        max_link_uti = 0
        min_link_uti = 0
        uti_std = 0

        while not done and step_count < EPISODE_LENGTH:
            action_dist, _ = agent.pred_action_distrib_sp(env, source, destination)
            action = np.argmax(action_dist)
            reward, done, error_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env.step(
                action, demand, source, destination)
            total_reward += reward
            step_count += 1
            max_link_uti = maxLinkUti[2] if len(maxLinkUti) > 2 else 0
            min_link_uti = minLinkUti
            uti_std = utiStd

        print(f" Hoàn thành: reward={total_reward:.4f}, steps={step_count}, errors={error_links}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        result_file = os.path.join(output_folder, f"{topology_name}_{tm_id}_result.pkl")
        result_data = {
            'tm_id': tm_id,
            'topology': topology_name,
            'reward': total_reward,
            'steps': step_count,
            'error_links': error_links,
            'max_link_uti': max_link_uti,
            'min_link_uti': min_link_uti,
            'uti_std': uti_std
        }

        with open(result_file, 'wb') as f:
            pickle.dump(result_data, f)

        print(f" Đã lưu kết quả vào: {result_file}")

    except Exception as e:
        print(f" Lỗi trong lúc đánh giá: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    K.clear_session()
    gc.collect()

    print(" Xong\n")
