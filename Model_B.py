# -*- coding: utf-8 -*-
import math
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import time
start_time = time.time()

population_size = 1000
generation_num = 3000
er = 0.1
cr = 0.4

rotor_diameter = 40
grid_size_ver = 1       # in diameter
grid_size_hor = 3       # in diameter
turbine_num = 30

wind_farm_size_hor = 2000
wind_farm_size_ver = 2000

row_num = math.ceil(wind_farm_size_ver / (grid_size_ver * rotor_diameter))
column_num = math.ceil(wind_farm_size_hor / (grid_size_hor * rotor_diameter))
cell_num = row_num * column_num

entrainment_constant = 0.09437
thrust_coefficient = 0.88
axial_ind_factor = 0.5 * (1 - (1 - thrust_coefficient) ** 0.5)

cut_power_coefficient = 0.99999
cut_in_wind_speed = 2       # in m/s
rated_wind_speed = 12.8     # in m/s
avg_wind_speed = 12         # in m/s
air_density = 1.2254        # in kg/m3
power_coefficient = 0.4
turbine_power = 0.5 * power_coefficient * air_density * math.pi * ((rotor_diameter ** 2) / 4) * (avg_wind_speed ** 3) / 1000  # in kW


def print_list(listo):
    print("========")
    for i in listo:
        print(i)
    print("========")


def is_value_between(val, minn, maxx):
    return val <= minn or val >= maxx


def is_static_values_valid():
    warning_msg_list = []
    if population_size < 10:
        msg = "population_size, 10 dan küçük olamaz."
        warning_msg_list.append(msg)
    if population_size % 10 != 0:
        msg = "population_size, 10 nun katı olmalı."
        warning_msg_list.append(msg)
    if generation_num < 3:
        msg = "generation_num, 3 den küçük olamaz."
        warning_msg_list.append(msg)
    if is_value_between(cut_power_coefficient, 0, 1):
        msg = "cut_power_coefficient 0 ile 1 arasında olmalı."
        warning_msg_list.append(msg)
#    if is_value_between(loss, 0, 1):
#        msg = "loss 0 ile 1 arasında olmalı."
#        warning_msg_list.append(msg)
    if is_value_between(er, 0, 1):
        msg = "er 0 ile 1 arasında olmalı."
        warning_msg_list.append(msg)
    if is_value_between(cr, 0, 1):
        msg = "cr 0 ile 1 arasında olmalı."
        warning_msg_list.append(msg)
    if is_value_between(er + cr, 0, 1):
        msg = "er + cr 0 ile 1 arasında olmalı."
        warning_msg_list.append(msg)
    if turbine_num > cell_num:
        msg = "turbine_num, cell_num dan büyük olamaz"
        warning_msg_list.append(msg)

    is_valid_values = len(warning_msg_list) == 0
    if not is_valid_values:
        print("----- WARNING -----")
        print_list(warning_msg_list)
        print("----- WARNING -----")
    return is_valid_values


def get_random_layout_raw_array():
    raw_array = np.hstack((np.ones((turbine_num,), int), np.zeros((cell_num - turbine_num,), int)))
    np.random.shuffle(raw_array)
    return raw_array


def get_column_divided_array_from_raw_array(raw_array):
    column_divided_array = []
    sub_array = []
    for cellIndex in range(len(raw_array)):
        sub_array.append(raw_array[cellIndex])
        if cellIndex % row_num == row_num - 1:
            column_divided_array.append(sub_array)
            sub_array = []
    return column_divided_array


def get_power_from_column_divided_array(column_divided_array):
    total_power = 0
    for column in column_divided_array:
        wind_speed = []
        for cellIndex in range(len(column)):
            counter = 1
            if column[cellIndex] == 1 and len(wind_speed) == 0:
                wind_speed.append(avg_wind_speed)
            elif column[cellIndex] == 1 and len(wind_speed) > 0:
                for row_number in range(cellIndex):
                    if cellIndex > 0 and column[cellIndex - row_number - 1] == 1:
                        reduced_speed = wind_speed[len(wind_speed)-1] * (1 - 2 * axial_ind_factor * ((rotor_diameter / 2) / (rotor_diameter / 2 + entrainment_constant * grid_size_ver * rotor_diameter * counter)) ** 2)
                        if reduced_speed < cut_in_wind_speed:
                            reduced_speed = 0
                        wind_speed.append(reduced_speed)
                        break
                    else:
                        counter += 1

        for i in range(len(wind_speed)):
            total_power = total_power + 0.5 * power_coefficient * air_density * math.pi * ((rotor_diameter ** 2) / 4) * (wind_speed[i] ** 3) / 1000
    return round(total_power, 4)


def get_layout_object_from_raw_array(raw_array):
    column_divided_array = get_column_divided_array_from_raw_array(raw_array)
    layout_obj = {
        "raw_array": raw_array,
        "column_divided_array": column_divided_array,
        "power": get_power_from_column_divided_array(column_divided_array)
    }
    return layout_obj


def get_random_layouts(size):
    layouts = []
    for i in range(size):
        layouts.append(get_layout_object_from_raw_array(get_random_layout_raw_array()))
    return layouts


def get_crossover_layouts(layouts):
    if len(layouts) % 2 != 0:
        return layouts
    else:
        ret_layouts = []
        for i in range(len(layouts)):
            if i % 2 == 1:
                raw_lay1 = layouts[i - 1]["raw_array"]
                raw_lay2 = layouts[i]["raw_array"]
                make_crossover(raw_lay1, raw_lay2)
                ret_layouts.append(get_layout_object_from_raw_array(raw_lay1))
                ret_layouts.append(get_layout_object_from_raw_array(raw_lay2))
        return get_mutated_layouts(ret_layouts)


def make_crossover(raw1, raw2):
    # print("CCCCCCC")
    # print(raw1)
    # print(raw2)
    raw1_valid_zeros_indexes = []
    raw1_valid_ones_indexes = []
    for i in range(len(raw1)):
        if (raw1[i] == 0) and (raw2[i] == 1):
            raw1_valid_zeros_indexes.append(i)
        elif (raw1[i] == 1) and (raw2[i] == 0):
            raw1_valid_ones_indexes.append(i)
    if len(raw1_valid_zeros_indexes) > 0:
        raw1_give_zero_index = random.choice(raw1_valid_zeros_indexes)
        raw1_give_one_index = random.choice(raw1_valid_ones_indexes)
        # print("raw1 give zero index: " + str(raw1_give_zero_index))
        # print("raw1 give one index: " + str(raw1_give_one_index))
        raw1[raw1_give_zero_index] = 1
        raw1[raw1_give_one_index] = 0
        raw2[raw1_give_zero_index] = 0
        raw2[raw1_give_one_index] = 1
    else:
        #print("crossover not valid")
        pass
    # print(raw1)
    # print(raw2)
    # print("CCCCCCC")


def get_mutated_layouts(layouts):
    ret_layouts = []
    for layout in layouts:
        raw_lay = layout["raw_array"]
        make_mutation(raw_lay)
        ret_layouts.append(get_layout_object_from_raw_array(raw_lay))
    return ret_layouts


def make_mutation(raw):
    # print("MMM")
    # print(raw)
    zeros_indexes = []
    ones_indexes = []
    for i in range(len(raw)):
        if raw[i] == 0:
            zeros_indexes.append(i)
        else:
            ones_indexes.append(i)
    toggle_zero_index = random.choice(zeros_indexes)
    toggle_one_index = random.choice(ones_indexes)
    raw[toggle_zero_index] = 1
    raw[toggle_one_index] = 0
    # print(raw)
    # print(toggle_zero_index)
    # print(toggle_one_index)
    # print("MMM")


def get_top_piece_generation(generation, rate):
    return copy.deepcopy(generation[0: int(population_size * rate)])


def get_generation(prev_generation):
    generation = []
    if len(prev_generation) == 0:
        generation += get_random_layouts(population_size)
    else:
        generation += get_top_piece_generation(prev_generation, er)
        generation += get_crossover_layouts(get_top_piece_generation(prev_generation, cr))
        generation += get_random_layouts(int(population_size * (1 - er - cr)))

    generation = sorted(generation, key=lambda k: k['power'], reverse=True)
    #print_list(generation)
    return generation


def start_evolution():
    winner_layouts_powers_list = []
    active_generation = get_generation([])
    winner_layouts_powers_list.append(active_generation[0]["power"])
    evolve_num = 0
    for i in range(generation_num):
        if active_generation[0].get("power") > cut_power_coefficient * turbine_num * turbine_power:
            break
        active_generation = get_generation(active_generation)
        winner_layouts_powers_list.append(active_generation[0]["power"])
        evolve_num += 1

    winner_lay = active_generation[0]
    execution_time = (time.time() - start_time)
    print("Execution time in sec: " + str(execution_time))
    print("-------------  Winner Layout  -------------")
    print("Evolve               : " + str(evolve_num) + " times")
    print("raw_array            : " + str(winner_lay["raw_array"]))
    print("column_divided_array : " + str(winner_lay["column_divided_array"]))
    print("power                : " + str(winner_lay["power"]))
    print("max power            : " + str(turbine_num * turbine_power))
    print("efficiency           : " + str(round(100 * winner_lay["power"] / (turbine_num * turbine_power), 2)) + " %")
    print("cutoff efficiency    : " + str(round(cut_power_coefficient * 100, 2)) + " %")
#    print("loss rate            : " + str(loss))
    print("-------------  Winner Layout  -------------")
    print("-------------  Winners List  -------------")
    print(len(winner_layouts_powers_list))
    print(winner_layouts_powers_list)
    print("-------------  Winners List  -------------")

    x = range(len(winner_layouts_powers_list))
    y = winner_layouts_powers_list
    plt.title("Line graph")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    # plt.ylim([0, turbine_num * turbine_power])
    plt.plot(x, y, color="red")
    plt.show()


if __name__ == '__main__':
    if is_static_values_valid():
        start_evolution()
