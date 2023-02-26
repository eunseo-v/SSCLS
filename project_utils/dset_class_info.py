# 存各类数据集的名称，便于画混淆矩阵

def pt_confmat_info_60():
    items_60 = ['drink water','eat meal/snack','brushing teeth','brushing hair','drop','pickup',
    'throw','sitting down','standing up (from sitting position)','clapping','reading','writing',
    'tear up paper','wear jacket','take off jacket','wear a shoe','take off a shoe','wear on glasses',
    'take off glasses','put on a hat/cap','take off a hat/cap','cheer up','hand waving','kicking something',
    'reach into pocket','hopping (one foot jumping)','jump up','make a phone call/answer phone',
    'playing with phone/tablet','typing on a keyboard','pointing to something with finger','taking a selfie',
    'check time (from watch)','rub two hands together','nod head/bow','shake head','wipe face','salute',
    'put the palms together','cross hands in front (say stop)','sneeze/cough','staggering','falling',
    'touch head (headache)','touch chest (stomachache/heart pain)','touch back (backache)','touch neck (neckache)',
    'nausea or vomiting condition','use a fan (with hand or paper)/feeling warm','punching/slapping other person',
    'kicking other person','pushing other person','pat on back of other person','point finger at the other person',
    'hugging other person','giving something to other person','touch other person_s pocket','handshaking',
    'walking towards each other','walking apart from each other']
    class_dict = dict(
        zip(range(60), items_60)
    )
    for key in class_dict.keys():
        class_dict[key] = '{}{}'.format(key+1, class_dict[key])
    return class_dict

def pt_confmat_info_10():
    items_10 = [
        'Preparation', 'Grasp Bird\'s tail', 'Single Whip',
        'Lift up Hand', 'White Crane Spread its Wings',
        'Brush Knee and Twist Step', 'Hold the Lute',
        'Pulling,Blocking and Pounding', 'Apparent Close Up',
        'Cross Hands'
    ]
    class_dict = dict(
        zip(range(10), items_10)
    )
    for key in class_dict.keys():
        class_dict[key] = '{}{}'.format(key+1, class_dict[key])
    return class_dict

def pt_confmat_info_6():
    items_6 = [
        'Vital signs measurements', 
        'Blood Collection', 
        'Blood Glucose Measurement',
        'Indwelling drip retention and connection', 
        'Oral care',
        'Diaper exchange and cleaning of area', 
    ]
    class_dict = dict(
        zip(range(6), items_6)
    )
    for key in class_dict.keys():
        class_dict[key] = '{}{}'.format(key+1, class_dict[key])
    return class_dict