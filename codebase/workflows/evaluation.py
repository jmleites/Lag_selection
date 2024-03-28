from neuralforecast.losses.numpy import smape


def calculate_smape(cv_df, model):
    smape_model = smape(y=cv_df['y'], y_hat=cv_df[model])
    return smape_model


def evaluation(evaluation_df, input_list):
    done = False
    best_input = []
    for model in range(len(evaluation_df)):
        done = False
        for i in range(len(evaluation_df.loc[model, 'best_model'])):
            letter = evaluation_df.loc[model, 'best_model'][i]
            if letter.isnumeric():
                if i < len(evaluation_df.loc[model, 'best_model']) - 1:
                    next_letter = evaluation_df.loc[model, 'best_model'][i + 1]
                    if next_letter.isnumeric():
                        numeric_str = letter + next_letter
                        best_input.append(int(numeric_str))
                        done = True
                        break
                    else:
                        best_input.append(int(letter))
                        done = True
                        break
                else:
                    best_input.append(int(letter))
                    done = True
        if not done:
            best_input.append(0)

    corresponding_values = []
    for i in best_input:
        corresponding_values.append(input_list[i])

    evaluation_df.insert(evaluation_df.columns.get_loc('best_model') + 1, 'best_input', corresponding_values)
    return evaluation_df
