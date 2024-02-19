import pickle, glob, os
from SQLDataModel import SQLDataModel

def get_pickle_file_names_from_dir(dir:str=None) -> list[str]:
    """Returns a list of file names found in `dir` matching pickle extension '.pkl', returning them as a list."""
    # Iterate through each subdirectory
    for subdir, _, _ in os.walk(dir):
        # Get list of pickle files in the current subdirectory
        pickle_files = glob.glob(os.path.join(subdir, '*.pkl'))
        # Print the pickle file names
    return [pkl for pkl in pickle_files]

def get_pickled_data(pickle_file:str) -> SQLDataModel:
    """Returns the specified `pickle_file` and returns a `SQLDataModel` object."""
    with open(pickle_file, 'rb') as f:
        obj = pickle.load(f) # Tuple of Tuples raw data
    headers = ['iteration', 'batch_idx', 'loss', 'avg batch loss', 'iteration_time']
    data = [[k, v[0], v[1], v[2], v[3]] for k,v in obj.items()]
    sdm = SQLDataModel(data, headers, display_float_precision=6)
    sdm.set_display_color('#A6D7E8')
    return sdm


data_dir = ['/home/cs744/hw2/DataFromOtherNodes/node0','/home/cs744/hw2/DataFromOtherNodes/node1','/home/cs744/hw2/DataFromOtherNodes/node2','/home/cs744/hw2/DataFromOtherNodes/node3']
sdm = None
for data_subdir in data_dir:
    all_data = get_pickle_file_names_from_dir(data_subdir)
    for pkl_file in all_data:
        # /home/cs744/hw2/DataFromOtherNodes/node3/part-3.pkl
        node_name = pkl_file.split('/')[-2]
        part_name = pkl_file.split('/')[-1]
        run_label = f"{node_name}-{part_name}"
        run_label = run_label.replace(".pkl","")
        res = get_pickled_data(pkl_file)
        res['label'] = run_label
        if 'part-1' in part_name:
            res['test accuracy'] = '2081/10000 (21%)'
        elif 'part-2a' in part_name:
            if 'node0' in node_name:
                res['test accuracy'] = '1001/10000 (10%)'
            elif 'node1' in node_name:
                res['test accuracy'] = '1003/10000 (10%)'
            elif 'node2' in node_name:
                res['test accuracy'] = '1001/10000 (10%)'
            elif 'node3' in node_name:
                res['test accuracy'] = '1001/10000 (10%)'
        elif 'part-2b' in part_name:
            if 'node0' in node_name:
                res['test accuracy'] = '1088/10000 (11%)'
            elif 'node1' in node_name:
                res['test accuracy'] = '1087/10000 (11%)'
            elif 'node2' in node_name:
                res['test accuracy'] = '1088/10000 (11%)'
            elif 'node3' in node_name:
                res['test accuracy'] = '1086/10000 (11%)'            
        elif 'part-3' in part_name:
            res['test accuracy'] = '1014/10000 (10%)'
        if sdm is None:
            sdm = res
        else:
            sdm = sdm.concat(res, inplace=False)
print(sdm.describe())
sdm.to_html('our-results.html')
sdm.to_csv('results.csv')

# results = get_pickled_data('part-2b.pkl')
# print(results)

"""
### part2a
python part2a_main.py -l 0
Average time per iteration for all 40 iterations: 3.4051441027193654
Test set: Average loss: 2.3076, Accuracy: 1001/10000 (10%)

python part2a_main.py -l 1
Average time per iteration for all 40 iterations: 3.4057859741911596
Test set: Average loss: 2.3076, Accuracy: 1003/10000 (10%)

python part2a_main.py -l 2
Average time per iteration for all 40 iterations: 3.4034725062701168
Test set: Average loss: 2.3075, Accuracy: 1001/10000 (10%)

python part2a_main.py -l 3
Average time per iteration for all 40 iterations: 3.4067170425337188
Test set: Average loss: 2.3075, Accuracy: 1001/10000 (10%)

### part2b
node 0 : 
Average time per iteration for all 40 iterations: 3.262401191555724
Test set: Average loss: 2.3204, Accuracy: 1088/10000 (11%)

node1: 
Average time per iteration for all 40 iterations: 3.2625135013035367
Test set: Average loss: 2.3170, Accuracy: 1087/10000 (11%)

node2: 
Average time per iteration for all 40 iterations: 3.2575074166667704
Test set: Average loss: 2.3193, Accuracy: 1088/10000 (11%)

node3: 
Average time per iteration for all 40 iterations: 3.2628387626336544
Test set: Average loss: 2.3178, Accuracy: 1086/10000 (11%)

### part3
node0:
Average time per iteration for all 40 iterations: 2.9484396224119225
Test set: Average loss: 2.4878, Accuracy: 1014/10000 (10%)

node1:
Average time per iteration for all 40 iterations: 2.948629763661599
Test set: Average loss: 2.4878, Accuracy: 1014/10000 (10%)

node2: 
Average time per iteration for all 40 iterations: 2.9476551814955108
Test set: Average loss: 2.4878, Accuracy: 1014/10000 (10%)

node3:
Average time per iteration for all 40 iterations: 2.9463119652806498
Test set: Average loss: 2.4878, Accuracy: 1014/10000 (10%)

"""