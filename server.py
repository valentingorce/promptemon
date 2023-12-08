from flask import Flask, render_template, request, url_for, session, redirect
import random
from utils import Promptemon, get_all_registered

app = Flask(__name__)
app.secret_key = "123"

@app.route('/')
def index():
    promptemons = get_all_registered()
    return render_template('index.html',
                           promptemons = promptemons)

@app.route('/generate', methods=['GET','POST'])
def generate():
    global foe, ally
    description = request.form.get('description')
    ally = Promptemon(description, gen_img=True)
    ally.save()
    img_path = ally.img_path
    # stats = {'type1':'Fire',
    #          'type2':'Dragon',
    #          'attack':80,
    #          'defense':70,
    #          'hp':90,
    #          'speed':85,
    #          'spe_atk':90,
    #          'spe_def':75,
    #          'skills':[{'name': 'Flamethrower', 'damage': 80, 'type': 'Fire', 'kind': 'special'}, 
    #                    {'name': 'Dragon Claw', 'damage': 75, 'type': 'Dragon', 'kind': 'physical'}, 
    #                    {'name': 'Fire Spin', 'damage': 70, 'type': 'Fire', 'kind': 'special'}, 
    #                    {'name': 'Wing Attack', 'damage': 65, 'type': 'Flying', 'kind': 'physical'}]}
    stats = ally.base_stats
    session['ally_promptemon'] = {'remaining_hp':ally.base_stats.hp, 
                                  'max_hp':ally.base_stats.hp, 
                                  'skills':stats.skills, 
                                  'img_path':img_path}
    foe = random.choice(get_all_registered())
    print('Selected foe: ', foe.u_id)
    session['foe_promptemon'] = {'remaining_hp':foe.base_stats.hp, 
                                 'max_hp':foe.base_stats.hp, 
                                 'img_path':foe.img_path}
    return render_template('new_promptemon.html',
                           img_path=img_path,
                           stats=stats)

@app.route('/pick', methods=['POST'])
def pick():
    global foe, ally
    picked_one_id = request.form.get('picked_one')
    print(picked_one_id)
    ally = Promptemon.load(picked_one_id)
    session['ally_promptemon'] = {'remaining_hp':ally.base_stats.hp, 
                                  'max_hp':ally.base_stats.hp, 
                                  'skills':ally.base_stats.skills, 
                                  'img_path':ally.img_path}
    
    foe = random.choice(get_all_registered())
    print('Selected foe: ', foe.u_id)
    session['foe_promptemon'] = {'remaining_hp':foe.base_stats.hp, 
                                 'max_hp':foe.base_stats.hp, 
                                 'img_path':foe.img_path}
    return render_template('new_promptemon.html',
                           img_path=ally.img_path,
                           stats=ally.base_stats)

@app.route('/battle', methods=['GET'])
def battle():
    session['ally_promptemon'] = {'remaining_hp':ally.remaining_hp, 
                                  'max_hp':ally.base_stats.hp, 
                                  'skills':ally.base_stats.skills, 
                                  'img_path':ally.img_path}
    session['foe_promptemon'] = {'remaining_hp':foe.remaining_hp, 
                                 'max_hp':foe.base_stats.hp, 
                                 'img_path':foe.img_path}
    return render_template('battle.html',
                           ally=session['ally_promptemon'],
                            foe=session['foe_promptemon']
                           )

@app.route('/attack', methods=['POST'])
def attack():
    global ally, foe
    skill_choice = request.form.get('skillsChoice')
    print('skill_choice: ', skill_choice)
    skill = ally.attack(skill_choice)
    print('skill:', skill)
    foe.defend(skill['type'],skill['true_damage'],skill['kind'])
    if foe.is_KO():
        return render_template('win.html')
    else:
        skill_choice = random.choice(foe.skills)
        print('foe skill_choice: ', skill_choice)
        skill = foe.attack(skill_choice)
        print('foe skill:', skill)
        ally.defend(skill['type'],skill['true_damage'],skill['kind'])
        if ally.is_KO():
            return render_template('lose.html')
        else:
            print(ally.remaining_hp)
            print(foe.remaining_hp)
            return redirect(url_for('battle'))

if __name__ == '__main__':
    app.run()