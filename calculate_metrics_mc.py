import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file_name', type=str, default='./Chinese_LLMs_outputs/multiple_choice/chatglm_pro_output.json')
    return parser.parse_args()


def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def calculate_acc(predicts, ground_truth):
    correct_count = 0
    for i in range(len(predicts)):
        correct_choice = ground_truth[i]["answer"][len('Answer: '):].strip()
        response = predicts[i]['response'].strip()
        if response.startswith('Answer: '):
            if response[len('Answer: '):] == correct_choice:
                correct_count += 1
        elif len(response) == 1 and response.isalpha():
            if response == correct_choice:
                correct_count += 1
    return correct_count / len(predicts)

if __name__ == '__main__':
    args = get_args()
    predicts = load_data(args.response_file_name)
    ground_truth = load_data('HalluQA_mc.json')
    print('Acc: {:.2f}%'.format(100 * calculate_acc(predicts, ground_truth)))