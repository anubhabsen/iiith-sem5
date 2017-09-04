import math
import pandas as pd
import sys
from pandas import DataFrame
import collections

MIN = float("inf")
debug = 0

train_data = DataFrame.from_csv(sys.argv[1], index_col = None)

train_data.loc[:,'satisfaction_level'] *= 200
train_data.loc[:,'satisfaction_level'] /= 20
train_data.loc[:,'satisfaction_level'] = train_data.loc[:,'satisfaction_level'].astype(int)

train_data.loc[:,'last_evaluation'] *= 200
train_data.loc[:,'last_evaluation'] /= 20
train_data.loc[:,'last_evaluation'] = train_data.loc[:,'last_evaluation'].astype(int)

train_data.loc[:,'average_montly_hours'] /= 15

train_data.loc[:,'average_montly_hours'] = train_data.loc[:,'average_montly_hours'].astype(int)

def entropy(probs):
    total = 0
    total = total + 0.0
    for prob in probs:
        if prob != 0:
            total += - prob * math.log(prob, 2)
        else:
            pass
    return total

def entropy_of_list(a_list):
    cnt = collections.Counter(x for x in a_list)
    num_instances = len(a_list)*1.0
    num_instances = num_instances + 0.0
    probs = []
    for x in cnt.values():
        probs.append(x)
    return entropy(probs)

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index) * 1.0
    aggregate = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
    aggregate.columns = ['Entropy', 'PropObservations']

    # new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    # old_entropy = entropy_of_list(df[target_attribute_name])
    answer = entropy_of_list(df[target_attribute_name]) - sum( aggregate['Entropy'] * aggregate['PropObservations'] )
    return answer

def id3(df, target_attribute_name, attribute_names, default_class=None):
    cnt = collections.Counter(x for x in df[target_attribute_name])
    if df.empty or (not attribute_names):
        return default_class
    elif len(cnt) == 1:
        return cnt.keys()[0]
    else:
        # Default Value for next recursive call
        block = max(cnt.values())
        index_of_max = cnt.values().index(block)
        default_class = cnt.keys()[index_of_max] # most common value of target attribute

        # Best Attribute
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        # index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[gainz.index(max(gainz))]

        # Empty tree
        tree = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree

def classify(instance, tree, default=None):
    # attribute = tree.keys()[0]
    if instance[tree.keys()[0]] in tree[tree.keys()[0]].keys():
        result = tree[tree.keys()[0]][instance[tree.keys()[0]]]
        if not isinstance(result, dict):
            return result
        else:
            x = classify(instance, result)
            return x
    else:
        # if default == None:
            # print(attribute, instance[attribute])
        return default


attribute_names = list(train_data.columns)
attribute_names.remove('left')

if debug == 0:
    test = DataFrame.from_csv(sys.argv[2], index_col = None)
    test.loc[:,'satisfaction_level'] *= 200
    test.loc[:,'satisfaction_level'] /= 20
    test.loc[:,'satisfaction_level'] = test.loc[:,'satisfaction_level'].astype(int)

if debug == 0:
    test.loc[:,'last_evaluation'] *= 200
    test.loc[:,'last_evaluation'] /= 20
    test.loc[:,'last_evaluation'] = test.loc[:,'last_evaluation'].astype(int)

    test.loc[:,'average_montly_hours'] /= 15
    test.loc[:,'average_montly_hours'] = test.loc[:,'average_montly_hours'].astype(int)

train_tree = id3(train_data, 'left', attribute_names)
test['predicted'] = test.apply(classify, axis=1, args=(train_tree,0) )
test.fillna(0, inplace=True)

# training_data = train_data
# test_data  = test

train_tree = id3(train_data, 'left', attribute_names)
test['predicted2'] = test.apply(classify, axis=1, args=(train_tree,0) )
test.fillna(0, inplace=True)

for i in test['predicted']:
    res = int(i)
    print(res)

# print 'Accuracy is ' + str( sum(test['left']==test['predicted2'] ) / (1.0*len(test.index)) )
