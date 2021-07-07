from priceRecord import PriceRecord
from datetime import datetime
from datetime import date
from datetime import timedelta
import csv
import numpy as np


def get_price_history(file_name) -> list:
    price_history = []
    with open('./csv_files/' + file_name) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            record_date = datetime.strptime(row[0], '%Y.%m.%d').date()  # type is 'datetime.date'
            # print(type(date))
            end = float(row[5])
            price_record = PriceRecord(record_date, end)
            price_history.append(price_record)
    return price_history


def add_years(day, num_years) -> date:
    try:
        return day.replace(year=day.year + num_years)
    except ValueError:
        return day + (date(day.year + num_years, 1, 1) - date(day.year, 1, 1))


def add_day(day) -> date:
    return day + timedelta(days=1)


def calculate_values(first_day, last_day) -> list:
    num_of_products = len(price_records)
    indices_of_first_day = [0] * num_of_products
    for product in range(num_of_products):
        current_index = 0
        while (price_records[product][current_index].date < first_day and
               current_index < len(price_records[product]) - 1):  # why need minus 1
            current_index += 1
        indices_of_first_day[product] = current_index

    values = []  # consider renaming this and quantities
    quantities = list(map(lambda product_num: portfolio_product_ratio[product_num] /
                          sum(portfolio_product_ratio) /
                          price_records[product_num][indices_of_first_day[product_num]].end,
                          range(num_of_products)))

    indices_of_current_day = [0] * num_of_products
    d = first_day
    while d < last_day:
        for product in range(num_of_products):
            current_index = 0
            while (price_records[product][current_index].date < d and
                   current_index < len(price_records[product]) - 1):
                current_index += 1
            indices_of_current_day[product] = current_index
        values.append(sum(list(map(
            lambda product_num: quantities[product_num] *
            price_records[product_num][indices_of_current_day[product_num]].end,
            range(num_of_products)))))
        d = add_day(d)  # see if the logic is correct but should be okay
    return values


price_records = [
    get_price_history("US_5001440.csv"),
    get_price_history("CHINA_A501440.csv"),
    get_price_history("DAX301440.csv"),
    get_price_history("GOLD1440.csv"),
    get_price_history("SILVER1440.csv"),
]

earliest_date_list = list(map(lambda product_price_history: product_price_history[0].date,
                              price_records))
earliest_date = max(earliest_date_list)
latest_date_list = list(map(lambda product_price_history: product_price_history[-1].date,
                            price_records))
latest_date = min(latest_date_list)

values_on_last_day = []
random_numbers = [float(line.strip()) for line in open('random_num.txt', 'r').readlines()]
portfolio_product_ratio = [1, 1, 1, 1, 0]

for i in range(300):
    random_num = random_numbers[i]
    random_date = earliest_date + (latest_date - earliest_date) * random_num
    # print(random_date, add_year(random_date))
    year_values = calculate_values(random_date, add_years(random_date, 1))
    values_on_last_day.append(year_values[-1])

print(np.average(values_on_last_day))
print(np.percentile(values_on_last_day, 1))



