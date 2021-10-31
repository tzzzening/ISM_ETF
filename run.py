from priceRecord import PriceRecord
from datetime import datetime
from datetime import date
from datetime import timedelta
import csv
import numpy as np
import time
import math
import copy
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from numpy.random import randint
from random import sample
import random


def get_price_history(file_name) -> list:
    price_history = []
    with open('./csv_files/Stashaway/' + file_name, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            record_date = datetime.strptime(row[0], '%d-%m-%y').date()  # type is 'datetime.date'
            # record_date = datetime.strptime(row[0], '%Y.%m.%d').date()
            try:
                end = float(row[1])
            except ValueError:
                continue
            price_record = PriceRecord(record_date, end)
            price_history.append(price_record)
    return price_history


def add_time(initial_day, num_units, unit) -> date:
    if unit == 'year':
        if num_units > 0:
            return add_years(add_months(initial_day, round((num_units % 1) * 12)), math.floor(num_units))
        else:
            return add_years(add_months(initial_day, -round(((-num_units) % 1) * 12)), math.ceil(num_units))
    elif unit == 'month':
        return add_months(initial_day, num_units)
    elif unit == 'day':
        return add_day(initial_day, num_units)
    else:
        raise TypeError


def add_years(day, num_years) -> date:
    try:
        return day.replace(year=day.year + num_years)
    except ValueError:
        return day + (date(day.year + num_years, 1, 1) - date(day.year, 1, 1))


def add_months(day, num_months) -> date:
    return day + timedelta(days=num_months * 30)


def add_day(initial_day, num_days) -> date:
    return initial_day + timedelta(days=num_days)


def calculate_values(first_day, last_day, records, portfolio_product_ratio) -> list:
    num_of_products = len(records)
    indices_of_first_day = [0] * num_of_products
    for product in range(num_of_products):
        current_index = 0
        while records[product][current_index].date < first_day:
            current_index += 1
        indices_of_first_day[product] = current_index
        # locates the index / position of the first day in that product's history

    daily_ratio_against_first_day_price = []
    fraction_of_product_in_portfolio = list(map(lambda product_num: portfolio_product_ratio[product_num] /
                                                                    sum(portfolio_product_ratio),
                                                range(num_of_products)))

    indices_of_current_day = [0] * num_of_products

    for product in range(num_of_products):
        current_index = 0
        try:
            while records[product][current_index].date < last_day:
                current_index += 1
        except IndexError:
            current_index = -1
        indices_of_current_day[product] = current_index
    daily_ratio_against_first_day_price = sum(list(map(
        lambda product_num: fraction_of_product_in_portfolio[product_num] *
                            records[product_num][indices_of_current_day[product_num]].end /
                            records[product_num][indices_of_first_day[product_num]].end,
        range(num_of_products))))

    return daily_ratio_against_first_day_price


def split_data(price_records, train_latest_date, test_latest_date) -> tuple:
    train_records = []
    test_records = []
    for product in price_records:
        i = len(product) - 1
        while True:
            if product[i].date <= train_latest_date:
                break
            i -= 1

        j = len(product) - 1
        while True:
            if product[j].date <= test_latest_date:
                break
            j -= 1

        print(product[j].date)
        train_records.append(product[:i + 1])
        test_records.append(product[i + 1: j + 1])
    return train_records, test_records


def generate_initial_state():
  rand_list = [rand() for i in range(k_dim)]
  summ = np.sum(rand_list)
  rand_list = list(map(lambda i: i / summ, rand_list))
  return rand_list


def generate_new_state(portfolio_product_ratio, step_size, num_assets_to_switch):
    rand_indices = sample(range(k_dim), num_assets_to_switch)
    switch_list = [portfolio_product_ratio[i] for i in rand_indices]
    max_percentage_out_of_range = 0
    sum_numbers_to_switch = sum(switch_list)
    change_list = generate_random_numbers_for_switch(num_assets_to_switch, step_size)
    temp_new_ratios = list(np.array(switch_list) + np.array(change_list))
    for i in range(num_assets_to_switch):
        percentage_out_of_range = max(-temp_new_ratios[i] / abs(change_list[i]),
                                      (temp_new_ratios[i] - sum_numbers_to_switch) / abs(change_list[i]))
        max_percentage_out_of_range = max(percentage_out_of_range, max_percentage_out_of_range)
    if max_percentage_out_of_range > 0:
        change_list = [x * (1 - max_percentage_out_of_range) for x in change_list]
        temp_new_ratios = list(np.array(change_list) + np.array(switch_list))

    for i in range(num_assets_to_switch):
        portfolio_product_ratio[rand_indices[i]] = temp_new_ratios[i]

    return portfolio_product_ratio


def generate_random_numbers_for_switch(num_assets_to_switch, step_size):
    rand_list = [random.random() for _ in range(num_assets_to_switch)]
    mean_diff = np.mean(rand_list)
    rand_list = [(x - mean_diff) for x in rand_list]
    max_num = max([abs(x) for x in rand_list])
    rand_tune = random.uniform(0.8, 1.0)
    rand_list = [x / max_num * rand_tune * step_size for x in rand_list]
    return rand_list


def prepare_data(file_names):
    price_records = [get_price_history(i) for i in file_names]

    earliest_date_list = list(map(lambda product_price_history: product_price_history[0].date,
                                  price_records))
    train_earliest_date = max(earliest_date_list)
    latest_date_list = list(map(lambda product_price_history: product_price_history[-1].date,
                                price_records))
    test_latest_date = min(latest_date_list)
    # test_latest_date = min(date(2021, 6, 30), test_latest_date)
    train_latest_date = add_time(test_latest_date, -test_period, 'year')
    print(train_latest_date)
    test_earliest_date = add_time(train_latest_date, 1, 'day')

    train_records, test_records = split_data(price_records, train_latest_date, test_latest_date)
    return train_records, test_records, train_earliest_date, train_latest_date, test_earliest_date, test_latest_date


def objective(portfolio_product_ratio, current_iteration, y_intercept_penalty, temp_penalty_scores) -> tuple:
    train_values_on_last_day = []
    # random_numbers = [float(line.strip()) for line in open('random_num.txt', 'r').readlines()]
    for i in range(300):
        # for i in range(5485):
        # for i in range(7787):
        # for i in range(5):
        random_num = random_numbers[i]
        random_start_date = train_earliest_date + (
                    add_time(train_latest_date, -sample_period, 'year') - train_earliest_date) * random_num
        random_end_date = add_time(random_start_date, sample_period, 'year')
        year_values = calculate_values(random_start_date, random_end_date, train_records, portfolio_product_ratio)
        train_values_on_last_day.append(year_values)

    average_return = np.average(train_values_on_last_day)
    risk = 1 - np.percentile(train_values_on_last_day, 1)

    if model_num == 1 or model_num == 3:
        fixed_penalty = 1
        is_risk_acceptable = int(risk <= risk_tolerance)
        obj_val = average_return * is_risk_acceptable

    elif model_num == 2:
        penalty_scores_scale = 1
        risk_penalty = (risk - risk_tolerance) * (
                    y_intercept_penalty - math.log(n_iterations - current_iteration + 1)) * penalty_scores_scale
        risk_penalty = max(0, risk_penalty)
        # print('risk - risk_tolerance', risk - risk_tolerance)
        # print('(y_intercept_penalty - math.log(n_iterations - current_iteration + 1)) * penalty_scores_scale', (y_intercept_penalty - math.log(n_iterations - current_iteration + 1)) * penalty_scores_scale)
        # print('risk_penalty', risk_penalty)
        # print(y_intercept_penalty - math.log(n_iterations - current_iteration + 1))
        # 1 to prevent it from going to 0. will have error with math.log(0)
        temp_penalty_scores.append(risk_penalty)
        obj_val = average_return - risk_penalty
    else:
        raise TypeError
    print('obj_val, average_return, risk', obj_val, average_return, risk)

    return obj_val, average_return, risk


def simulated_annealing(objective, n_iterations, starting_step_size, temp, k_dim):
    y_intercept_penalty = math.log(n_iterations + 1)
    temp_penalty_scores = []
    best_ratio_list = []
    best_obj_val_list = []
    best_returns_list = []
    returns_list = []
    risk_list = []

    best = generate_initial_state()
    best_obj_val, best_returns, best_risk = objective(best, 1, y_intercept_penalty, temp_penalty_scores)
    print(best_obj_val, best_returns, best_risk)
    # while loop ensures that risk is not higher than 10%
    while best_risk > risk_tolerance:
        # print('finding best')
        # best = rand() * (bounds[1] - bounds[0])
        # portfolio_product_ratio = [best, 1 - best]
        best = generate_initial_state()
        best_obj_val, best_returns, best_risk = objective(best, 1, y_intercept_penalty, temp_penalty_scores)

    curr = copy.deepcopy(best)
    curr_obj_val, curr_returns, curr_risk = best_obj_val, best_returns, best_risk

    for i in range(n_iterations):
        print('\nround', i)
        if not is_step_size_constant:
            step_size = exp(-float(i) / step_size_constant) * starting_step_size  # not sure if should use temp
        elif is_step_size_constant:
            step_size = starting_step_size

        candidate = generate_new_state(curr, step_size, num_assets_to_switch)
        print(candidate)
        candidate_obj_val, candidate_returns, candidate_risk = objective(candidate, i, y_intercept_penalty,
                                                                         temp_penalty_scores)  # need to change the + 0.5

        diff = candidate_obj_val - curr_obj_val
        t = temp / float(i + 1)
        metropolis = exp(-diff / t)
        if diff >= 0 or rand() < metropolis:
            curr = copy.deepcopy(candidate)
            curr_obj_val, curr_returns, curr_risk = candidate_obj_val, candidate_returns, candidate_risk

        if candidate_obj_val > best_obj_val:
            # if candidate_obj_val > best_obj_val and candidate_risk <= risk_tolerance: # dk why this doesn't work
            best = copy.deepcopy(candidate)
            best_obj_val, best_returns, best_risk = candidate_obj_val, candidate_returns, candidate_risk
            print('>%d f(%s) = %.5f' % (i, best, best_obj_val))

        best_ratio_list.append(best)
        best_obj_val_list.append(best_obj_val)
        best_returns_list.append(best_returns)
        returns_list.append(candidate_returns)
        risk_list.append(candidate_risk)

    with open('./data/temp.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['best_ratio_list', 'best_obj_val_list', 'best_returns_list', 'returns_list', 'risk_list', 'model_num',
             model_num, 'step_size', starting_step_size, 'switches', num_assets_to_switch, 'seed', seed_num])
        for i in range(n_iterations):
            writer.writerow(
                [best_ratio_list[i], best_obj_val_list[i], best_returns_list[i], returns_list[i], risk_list[i]])

    return [best, best_obj_val, best_returns, best_risk]


program_start_time = time.time()

file_names = [
        "stashaway_BNDX.MX_trimmed.csv",
        "stashaway_EMB_trimmed.csv",
        # "stashaway_EWJ_trimmed.csv",
        "stashaway_FLOT_trimmed.csv",
        "stashaway_GLD_trimmed.csv",
        "stashaway_IGOV_trimmed.csv",
        "stashaway_IJR_trimmed.csv",
        "stashaway_KWEB_trimmed.csv",
        "stashaway_TIP_trimmed.csv",
        # "stashaway_XLE_trimmed.csv",
        # "stashaway_XLP_trimmed.csv",
        "stashaway_XLV_trimmed.csv",

        # "stashaway_BNDX.MX_trimmed.csv",
        # "stashaway_EMB_trimmed.csv",
        # # "stashaway_EWJ_trimmed.csv",
        # "stashaway_FLOT_trimmed.csv",
        # "stashaway_GLD_trimmed.csv",
        # "stashaway_IGOV_trimmed.csv",
        # "stashaway_IJR_trimmed.csv",
        # "stashaway_KWEB_trimmed.csv",
        # "stashaway_TIP_trimmed.csv",
        # # "stashaway_XLE_trimmed.csv",
        # # "stashaway_XLP_trimmed.csv",
        # "stashaway_XLV_trimmed.csv",
        # "US_5001440_date_end.csv",
        # "CHINA_A501440_date_end.csv",
        # "DAX301440_date_end.csv",
    #     # "GOLD1440_date_end.csv",
    #     # "SILVER1440_date_end.csv",
        # "ARKG_date_end.csv",
        # "ARKK_date_end.csv",
        # "QQQ_date_end.csv",
        # "SCHA_date_end.csv",
        # "VUG_date_end.csv",
    ]

random_numbers = [float(line.strip()) for line in open('random_num.txt', 'r').readlines()]
seed_num = 3
random.seed(seed_num) ## EDIT
seed(seed_num) ## EDIT
num_assets_to_switch = 2  ## EDIT
starting_step_size = 0.5  ## EDIT
model_num = 2  ## EDIT
n_iterations = 300
step_size_constant = 50
temp = 10
k_dim = len(file_names)
risk_tolerance = 0.1
test_period = 2
sample_period = 1
if model_num == 1 or model_num == 2:
    is_step_size_constant = True
elif model_num == 3:
    is_step_size_constant = False
train_records, test_records, train_earliest_date, train_latest_date, test_earliest_date, test_latest_date = prepare_data(file_names)
best_product_portfolio_ratio, score, score_returns, score_risk = simulated_annealing(objective, n_iterations, starting_step_size, temp, k_dim)

stashaway_values_on_last_day = []
stashaway_portfolio_product_ratio = [20,10,4,19,20,5,4,6,11]

for i in range(300):
    random_num = random_numbers[i]
    random_start_date = test_earliest_date + (add_time(test_latest_date, -sample_period, 'year') - test_earliest_date) * random_num
    random_end_date = add_time(random_start_date, sample_period, 'year')
    year_values = calculate_values(random_start_date, random_end_date, test_records, stashaway_portfolio_product_ratio)
    stashaway_values_on_last_day.append(year_values)

stashaway_average_return = np.average(stashaway_values_on_last_day)
stashaway_risk = 1 - np.percentile(stashaway_values_on_last_day, 1)

test_values_on_last_day = []

for i in range(300):
    random_num = random_numbers[i]
    random_start_date = test_earliest_date + (add_time(test_latest_date, -sample_period, 'year') - test_earliest_date) * random_num
    random_end_date = add_time(random_start_date, sample_period, 'year')
    year_values = calculate_values(random_start_date, random_end_date, test_records, best_product_portfolio_ratio)
    test_values_on_last_day.append(year_values)

test_average_return = np.average(test_values_on_last_day)
test_risk = 1 - np.percentile(test_values_on_last_day, 1)


print('TRAIN')
for i in range(k_dim):
    print(best_product_portfolio_ratio[i])

print(score)
print(score_returns)
print(score_risk)

# print('TEST')
print(stashaway_average_return)
print(stashaway_risk)
print(test_average_return)
print(test_risk)