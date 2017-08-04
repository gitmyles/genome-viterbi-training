#Myles' Viterbi Algorithm for HMM
#UD CISC636 - Dr. Li Liao - Homework #2 - Due 10/25/16
#Code written entirely by Myles Johnson-Gray (mjgray@udel.edu)

#This system uses HMM and Viterbi Training to find the most likely path for a given sequence.
#We train on the training sequences and report the evaluation on the testing sequence.

import math #used for log()
#----------------------------------------------------------------------------------------------------------------------

# DYNAMIC PROGRAMMING TABLE (print a table of steps from dictionary)
def dptable(V):
     yield " ".join(("%12d" % i) for i in range(len(V)))
     for state in V[0]:
         yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

#----------------------------------------------------------------------------------------------------------------------

# VITERBI ALGORITHM (return the most probable path given model parameters)
def viterbi(obs, states, start_p, trans_p, emit_p, empty_list):
    V = [{}]
    pred_states = empty_list #list to hold the strings of the most probable paths
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    #Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            #max_tr_prob = max(V[t-1][prev_st]["prob"]* trans_p[prev_st][st] for prev_st in states)
            max_tr_prob = max(V[t - 1][prev_st]["prob"] + (math.log(trans_p[prev_st][st])*.0001) for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"]  + (math.log(trans_p[prev_st][st])*.0001) == max_tr_prob:
                    #max_prob = max_tr_prob * emit_p[st][obs[t]]    #should be addition
                    max_prob = (max_tr_prob) + (math.log(emit_p[st][obs[t]])*.0001)
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    for line in dptable(V):
        print(line)
    opt = [] #list containing the optimal path
    #The highest probability.
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    #Get most probable state and its backtrack.
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    #Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    #Output information to the user and create optimal path string (pred_states)
    print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)
    print('')
    pred_states = pred_states.append(''.join(opt))

#----------------------------------------------------------------------------------------------------------------------

#INITIALIZATION (initialize initial and model parameters)
states = ('+', '-')
observations = ("GAGTCTGCGATAAGCCCCGAACATATGCGATAAGCAAAAATGGCGGACTGCCTTGTTAGGGGTCTCAGGAAAACTTTGTTTGCAAGTTTTGCGGCTAGCGA")
start_probability = {'+': 0.5, '-': 0.5}
transition_probability = {
    '+' : {'+': 0.7, '-': 0.3},
    '-' : {'+': 0.4, '-': 0.6}
    }
emission_probability = {
    '+' : {'A': 0.5, 'C': 0.2, 'G': 0.2, 'T': 0.1},
    '-' : {'A': 0.1, 'C': 0.3, 'G': 0.1, 'T': 0.5}
    }

#Open training data and read line-by-line into content.
fname = "hw2_train.data"
with open(fname) as f:
    content = f.readlines()

#Open testing data and read line-by-line into content.
fname2 = "hw2_test.data"
with open(fname2) as f:
    content2 = f.readlines()

#Read in the TRAINING text file in FASTA format.
flip = 0 #determines whether assigning to state or sequence
i=0
j=0
sequence = [] #holds all the input sequences
state = [] #holds the states for the input sequences
for lines in content:
    if ">" not in lines: #if line doesn't contain >, than it's useful
        if flip ==0:
            sequence.append(lines.strip('\n'))
            flip = 1
        else:
            state.append(lines.strip('\n'))
            flip =0

# Read in the TESTING text file in FASTA format.
flip = 0  # determines whether assigning to state or sequence
i = 0
j = 0
test_sequence = []  # holds all the input sequences
test_state = []  # holds the states for the input sequences
for lines in content2:
    if ">" not in lines:  # if line doesn't contain >, than it's useful
        if flip == 0:
            test_sequence.append(lines.strip('\n'))
            flip = 1
        else:
            test_state.append(lines.strip('\n'))
            flip = 0

#----------------------------------------------------------------------------------------------------------------------

#MAIN EXECUTION (Perform training on the training set and test on the test set using Viterbi approach)

iterations = 3 #controls the number of iterations to run the Viterbi training
for h in range(0,iterations):
    pred_states = []
    for i in range(0,len(sequence)): #iterate through all training sequences
        #Perform Viterbi on the training sequences.
        viterbi(sequence[i], states, start_probability, transition_probability, emission_probability, pred_states)

    #Initialize variables to calculate new model parameters
    gene_count = 0 #count of "+"
    intergenic_count = 0 #count of "-"
    emission_gene = {'A':0, 'C':0, 'G':0, 'T':0} #count of genomes given "+" state
    emission_inter = {'A':0, 'C':0, 'G':0, 'T':0} #count of genomes given "-" state
    trans_gene_to = {'+':0, '-':0} #count of transitions from + (++ and +-)
    trans_inter_to = {'+':0, '-':0} #count of transitions from - (-+ and --)

    #Iterate through the number of solutions...
    for i in range(0, len(pred_states)):
        #Get the counts of states "+" and "-"
        gene_count += pred_states[i].count("+")
        intergenic_count += pred_states[i].count("-")

        #Depending on the state, increment the emission of a genome given the state.
        for k in range(0, len(pred_states[i])):
            if pred_states[i][k] == "+":
                emission_gene[sequence[i][k]] +=1
            else:
                emission_inter[sequence[i][k]] += 1

        #Increment the proper transition counter when a state follows a previous state.
        for k in range(1, len(pred_states[i])):
            if pred_states[i][k-1] == "+":
                trans_gene_to[pred_states[i][k]] +=1
            else:
                trans_inter_to[pred_states[i][k]] += 1

    #Change all emission counts that are "0" to some low value instead.
    for item in emission_gene:
        if emission_gene[item] == 0:
            emission_gene[item] += 10
        if emission_inter[item] == 0:
            emission_inter[item] += 10

    #Calculate new transition and emission probabilities.
    transition_probability["+"]["+"] = trans_gene_to["+"] / gene_count
    transition_probability["+"]["-"] = trans_gene_to["-"] / gene_count
    transition_probability["-"]["+"] = trans_inter_to["+"] / intergenic_count
    transition_probability["-"]["-"] = trans_inter_to["-"] / intergenic_count
    emission_probability["+"]["A"] = emission_gene["A"] / gene_count
    emission_probability["+"]["C"] = emission_gene["C"] / gene_count
    emission_probability["+"]["G"] = emission_gene["G"] / gene_count
    emission_probability["+"]["T"] = emission_gene["T"] / gene_count
    emission_probability["-"]["A"] = emission_inter["A"] / intergenic_count
    emission_probability["-"]["C"] = emission_inter["C"] / intergenic_count
    emission_probability["-"]["G"] = emission_inter["G"] / intergenic_count
    emission_probability["-"]["T"] = emission_inter["T"] / intergenic_count

    #Print resulting model parameters to the user.
    print("Iteration: " + str(h))
    print("Transistion(++,+-,-+,--): " + str(transition_probability["+"]["+"]) + " " + str(transition_probability["+"]["-"]) + " " +
          str(transition_probability["-"]["+"]) + " " +  str(transition_probability["-"]["-"]))
    print("Emission(+A,+C,+G,+T): " + str(emission_probability["+"]["A"]) + " " + str(emission_probability["+"]["C"]) + " " +
          str(emission_probability["+"]["G"]) + " " + str(emission_probability["+"]["T"]))
    print("Emission(-A,-C,-G,-T): " + str(emission_probability["-"]["A"]) + " " + str(emission_probability["-"]["C"]) + " " +
          str(emission_probability["-"]["G"]) + " " + str(emission_probability["-"]["T"]))
    print("#----------------------------------------------------------------------------------------------------------------------")
    print(test_sequence)
    print(test_state)

#NOW EVALUATE THE TEST SEQUENCE!!!

pred_states = []
viterbi(test_sequence[0], states, start_probability, transition_probability, emission_probability, pred_states)
#Initialize confusion matrix variables.
TP=0 #true positive
FP=0 #false positive
TN=0 #true negative
FN=0 #false negative

#Calculate confusion matrix variables.
for i in range(0, len(test_sequence)):
    for j in range(0, len(test_sequence[i])):
        if((test_state[i][j] is "+") & (pred_states[i][j] is "+")):
            TP+=1
        if ((test_state[i][j] is "-") & (pred_states[i][j] is "+")):
            FP += 1
        if ((test_state[i][j] is "+") & (pred_states[i][j] is "-")):
            TN += 1
        if ((test_state[i][j] is "-") & (pred_states[i][j] is "-")):
            FN += 1

#Calculate and print evaluation results to the user.
sensitivity = TP/(TP+FN)
specificity = TP/(TP+FP)
correlation = ((TP * TN) - (FP * FN)) / (math.sqrt(pred_states[0].count("+") * pred_states[0].count("-") * test_state[0].count("+") * test_state[0].count("-")))
print("Sensitivity: " + str(sensitivity) + "        Specificity: " + str(specificity) + "       Correlation Coefficient: " + str(correlation))