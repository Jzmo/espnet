import torch

idx_to_vq_min=[
    71,
    58,
    59,
    230,
    450,
    170,
    470,
    223,
    308,
    41,
    449,
    161,
    200,
    93,
    290,
    455,
    341,
    115,
    350,
    399,
    224,
    202,
    459,
    442,
    56,
    490,
    447,
    334,
    386,
    292,
    126,
    497,
    287,
    394,
    359,
    502,
    452,
    66,
    397,
    139,
    361,
    241,
    239,
    357,
    250,
    242,
    152,
    340,
    29,
    280,
    358,
    231,
    256,
    210,
    14,
    351,
    124,
    62,
    57,
    8,
    229,
    203,
    155,
    323,
    503,
    360,
    326,
    464,
    325,
    19,
    267,
    412,
    38,
    389,
    298,
    286,
    150,
    103,
    402,
    303,
    100,
    262,
    40,
    377,
    339,
    272,
    387,
    27,
    393,
    174,
    81,
    432,
    252,
    9,
    419,
    396,
    141,
    183,
    0,
    79,
    26,
    436,
    138,
    479,
    279,
    205,
    466,
    478,
    410,
    77,
    133,
    353,
    95,
]

idx_to_vq_max=[
    71,
    58,
    59,
    230,
    450,
    170,
    470,
    223,
    308,
    41,
    449,
    200,
    161,
    93,
    290,
    341,
    115,
    455,
    399,
    350,
    202,
    459,
    224,
    442,
    490,
    447,
    386,
    334,
    287,
    56,
    292,
    497,
    359,
    502,
    126,
    394,
    452,
    139,
    361,
    397,
    66,
    239,
    357,
    241,
    250,
    242,
    280,
    152,
    231,
    358,
    29,
    340,
    210,
    124,
    62,
    8,
    351,
    256,
    155,
    14,
    57,
    19,
    203,
    229,
    464,
    503,
    326,
    267,
    360,
    323,
    412,
    325,
    150,
    38,
    298,
    389,
    103,
    262,
    303,
    286,
    40,
    272,
    402,
    377,
    100,
    387,
    339,
    27,
    393,
    174,
    81,
    432,
    252,
    9,
    419,
    396,
    141,
    183,
    0,
    79,
    436,
    26,
    138,
    479,
    205,
    279,
    466,
    478,
    410,
    77,
    133,
    353,
    324,
    268,
    95,
    42
]

idx_to_vq = idx_to_vq_max

label_samples=[
    10888174,
    4239994,
    3764906,
    2832657,
    2355986,
    1407264,
    1465425,
    1283446,
    1309389,
    1198724,
    1076269,
    838217,
    770442,
    746908,
    675613,
    602303,
    599189,
    666542,
    564322,
    563007,
    491817,
    502596,
    573153,
    504542,
    489471,
    483707,
    443495,
    458990,
    417451,
    488922,
    395639,
    435645,
    420558,
    421057,
    454435,
    417093,
    394462,
    369486,
    370035,
    344336,
    373462,
    355162,
    337502,
    310203,
    322019,
    331866,
    280456,
    325277,
    278009,
    314916,
    314502,
    309215,
    271945,
    279476,
    268366,
    273357,
    282746,
    291919,
    277133,
    288849,
    280442,
    260575,
    278289,
    263387,
    260539,
    276940,
    265651,
    266467,
    268625,
    265779,
    259352,
    264321,
    233377,
    248513,
    249558,
    247734,
    238930,
    237114,
    237972,
    227185,
    233741,
    226440,
    230379,
    228354,
    223397,
    215155,
    218371,
    211654,
    216080,
    212899,
    208194,
    195754,
    190549,
    187444,
    177319,
    153806,
    145137,
    125141,
    124866,
    119975,
    104381,
    100514,
    84695,
    40134,
    21981,
    22182,
    8175,
    758,
    69,
    66,
    28,
    4,
    1,
    1,
    1,
    1,
]


# option 1: just give some specific weights
# normedWeights = [1 (x / sum(label_samples)) for x in label_samples]
# normedWeights[0:3] = [0.05, 0.08, 0.1]
# print('weights:', normedWeights)

# option 2: (log2(x+100)) ** 2  
normedWeights = torch.FloatTensor(label_samples)
normedWeights = torch.log2(normedWeights + 100)
normedWeights = normedWeights * normedWeights
normedWeights = 1 / (normedWeights / normedWeights.sum())
normedWeights = 100 * normedWeights / normedWeights.sum()

print('weights:', normedWeights)