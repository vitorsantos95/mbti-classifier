# Example of calculating the mcnemar test
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

dataset = pd.read_csv("resultados_v3.csv", sep='\t')
print(dataset.head())

languages = ["EN", "DE", "ES", "FR", "NL", "IT", "PT"]
classifiers = ["lstm", "BERT"]
dimensions = ["EI", "NS", "TF", "PJ"]

df = pd.DataFrame(columns=['idioma', 'type', 'chi', 'pvalue', 'alpha'])

for language in languages:
    for dimension in dimensions:
        query_bert = 'Idioma == "' + language + '" & type == "' + dimension + '" & classificador2 == "BERT"'
        query_lstm = 'Idioma == "' + language + '" & type == "' + dimension + '" & classificador2 == "lstm"'

        result_bert = dataset.query(query_bert)
        result_lstm = dataset.query(query_lstm)

        bert_correct = result_bert['Correct']
        bert_incorrect = result_bert['Incorrect']

        lstm_correct = result_lstm['Correct']
        lstm_incorrect = result_lstm['Incorrect']

        yes_yes = int(bert_correct) + int(lstm_correct)
        yes_no = int(bert_correct) + int(lstm_incorrect)
        no_yes = int(bert_incorrect) + int(lstm_correct)
        no_no = int(bert_incorrect) + int(lstm_incorrect)

        table = [[yes_yes, yes_no], [no_yes, no_no]]

        exact_value = False
        correction_value = False
        if (yes_yes < 25 or yes_no < 25 or no_yes < 25 or no_no < 25):
            exact_value = True
        else:
            exact_value = False
            correction_value = True

        # calculate mcnemar test
        result = mcnemar(table, exact=exact_value, correction=correction_value)
        alpha = 0.05

        chi = str('%.5f' % result.statistic)
        pvalue = str('%.5f' % result.pvalue)

        print(language + ";" + dimension + ";" + chi + ";" + pvalue + ";" + str(alpha))

    # interpret the p-value

    # if result.pvalue > alpha:
    #     print('Same proportions of errors (fail to reject H0)')
    # else:
    #     print('Different proportions of errors (reject H0)')
