import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score
from sklearn import svm
from joblib import dump, load
import logging
logging.basicConfig(level = logging.INFO)
import stanza
import itertools
import re


def get_halliday_dict(doc,nlp, get_words_not_found = False):
    """
    """

    doc = nlp(doc)

    result = {}
    resultado = []
    vocabulary_stats = {
        'verbs_not_found' : [],
        'verbs_found' : [],
        'adj_not_found':[],
        'adj_found':[]
    }

    for i in doc.sentences:
        words = i.words
        
        for w in words:
            new_w = {
                'id':w.id,
                'text':w.text,
                'lemma':w.lemma,
                'upos':w.upos,
                'head':w.head,
                'halliday_pos': None
            }

            for k in halliday_dict:
                if new_w['lemma'] in halliday_dict[k]:
                    new_w['halliday_pos'] = k
                    break
            

            vocabulary_stats['verbs_not_found'].append(new_w['lemma']) if new_w['halliday_pos'] == None and new_w['upos'] == 'VERB' else None
            vocabulary_stats['verbs_found'].append(new_w['lemma']) if new_w['halliday_pos'] != None and new_w['upos'] == 'VERB' else None

            new_w['halliday_pos'] = new_w['upos'] if new_w['halliday_pos'] == None else new_w['halliday_pos']
            resultado.append(new_w) 
    response = vocabulary_stats if get_words_not_found == True else resultado

    return response




def get_dependency_dict(words):
    result = {}
    pos_tagger = {}

    for w in words:
        pos_tagger[w['id']] = w['halliday_pos']
        result[w['head']] = []

    for k in words:
        try:
            result[k['head']].append(k['id'])
        except:
            pass

    return result, pos_tagger



def get_dependency_dict2(words):
    result = {}
    pos_tagger = {}

    for w in words:
        if w['lemma'] != 'no':
            pos_tagger[w['id']] = w['halliday_pos']
        else:
            pos_tagger[w['id']] = w['lemma']
        result[w['head']] = []

    for k in words:
        try:
            result[k['head']].append(k['id'])
        except:
            pass

    return result, pos_tagger



def get_patterns(dependency_dict, upos_dict, tripler = True):

    value = 2 if tripler else 1

    result = []
    new = []
    for k in dependency_dict:
        combinations = itertools.combinations(dependency_dict[k], value)
        for c in combinations:
            c += (k, )
            t = sorted(list(c))
            new_t = []
            for e in t:
                try:
                    current_pos = upos_dict[e]
                    new_t.append(current_pos) 
                except:
                    pass

            result.append(new_t) if len(new_t)>1 else None

    return result


def count_patterns_found(doc, pattern, nlp):

    words = get_halliday_dict(doc, nlp, False)
    dependency_dict, upos_dict = get_dependency_dict(words)
    triples = get_patterns(dependency_dict, upos_dict, True)
    double = get_patterns(dependency_dict, upos_dict, False)

    counter = triples.count(pattern) if len(pattern) == 3 else double.count(pattern)


    return counter




# Lista de sentimientos y verbos

tristeza = 'Abandono, Abatimiento, Abrumación, Aflicción, Agitación, Agobio, Agonía, Aislamiento, Amargura, Apatía, Arrepentimiento, Ausencia, Banalidad, Congoja, Consternación, Contrariedad, Contrición, Culpa, Decaimiento, Decepción, Dependencia, Depresión, Derrota, Desaliento, Desamor, Desamparo, Desánimo, Desaprobación, Desconsuelo, Descontento, Desdén, Desdicha, Desencanto, Desengaño, Desesperanza, Desgano, Desidia, Desilusión, Desmotivación, Desolación, Desprestigio, Desvalorización, Desventura, Devaluación, Disforia, Dolor, Duelo, Estancamiento, Exclusión, Fracaso, Humillación, Incapacidad, Incomprensión, Indiferencia, Inexpresividad, Infelicidad, Lástima, Malestar, Melancolía, Menosprecio, Necesidad, Neutralidad, Nostalgia, Pena, Perdición, Pesadumbre, Pesar, Pesimismo, Remordimiento, Resignación, Soledad, Sufrimiento, Suplicio, Tormento, Turbación, Vacío'
alegria = 'Alborozo, Alivio, Ánimo, Añoranza, Apaciguamiento, Arrojo, Asertividad, Autenticidad, Autocomplacencia, Autonomía, Bienaventuranza, Brillantez, Brío, Calma, Certeza, Comodidad, Complacencia, Contemplación, Contento, Deleite, Despreocupación, Dicha, Dignidad, Disfrute, Diversión, Ecuanimidad, Empoderamiento, Encanto, Entusiasmo, Esperanza, Euforia, Exaltación, Excitación, Éxito, Éxtasis, Fascinación, Felicidad, Fervor, Firmeza, Frenesí, Gozo, Grandeza, Gusto, Ilusión, Imperturbabilidad, Inspiración, Intrepidez, Jocosidad, Jolgorio, Jovialidad, Júbilo, Libertad, Logro, Lujuria, Motivación, Omnipotencia, Optimismo, Osadía, Ostentación, Pasión, Placer, Placidez, Plenitud, Regocijo, Revulsión, Satisfacción, Solemnidad, Sosiego, Suficiencia, Tranquilidad, Triunfo, Valentía, Vehemencia, Vigor, Vivacidad'
enfado = 'Abuso, Agravio, Agresividad, Arrogancia, Aspereza, Barbaridad, Beligerancia, Bravura, Brutalidad, Burla, Celo, Cólera, Coraje, Desconsideración, Desesperación, Despecho, Destrucción, Discordia, Disgusto, Displicencia, Dominación, Enajenamiento, Engaño, Engreimiento, Enjuiciamiento, Enojo, Envidia, Estrés, Exasperación, Fastidio, Ferocidad, Frustración, Furia, Hosquedad, Hostilidad, Impaciencia, Impotencia, Incomodidad, Inconformidad, Indignación, Injusticia, Insatisfacción, Insulto, Invasión, Ira, Irritabilidad, Malhumor, Manipulación, Molestia, Obligación, Obstinación, Odio, Orgullo, Pedantería, Petulancia, Prepotencia, Rabia, Rebeldía, Recelo, Rencor, Represión, Resentimiento, Superioridad, Traición, Ultranza, Venganza, Violencia'
amor = 'Aceptación, Acompañamiento, Admiración, Adoración, Afecto, Agradecimiento, Agrado, Amabilidad, Apego, Apoyo, Aprobación, Armonía, Atracción, Benevolencia, Bondad, Capricho, Cariño, Cercanía, Compasión, Comprensión, Compromiso, Condescendencia, Condolencia, Confianza, Conmiseración, Consideración, Consolación, Consuelo, Cordialidad, Correspondencia, Cuidado, Deseo, Dulzura, Embelesamiento, Empatía, Enamoramiento, Estima, Fortaleza, Generosidad, Gratitud, Heroísmo, Honestidad, Honorabilidad, Humildad, Integridad, Interés, Intimidad, Introspección, Justicia, Paciencia, Paz, Pertenencia, Receptividad, Respeto, Seguridad, Sensibilidad, Sensualidad, Sentimentalismo, Serenidad, Simpatía, Solidaridad, Solitud, Templanza, Tenacidad, Ternura, Tolerancia, Unidad, Valoración'
miedo = 'Achicamiento, Acobardamiento, Acoquinamiento, Alarma, Amilanamiento, Angustia, Ansiedad, Apocamiento, Aprensión, Canguelo, Cautela, Circunspección, Cobardía, Desasosiego, Desconfianza, Desprotección, Desvalimiento, Escrupulosidad, Espanto, Fobia, Fragilidad, Horror, Impasibilidad, Indefensión, Inestabilidad, Inferioridad, Inquietud, Inseguridad, Insignificancia, Insuficiencia, Intimidación, Intranquilidad, Medrosidad, Mezquindad, Mortificación, Nerviosismo, Pánico, Parálisis, Pavidez, Pavor, Perturbación, Preocupación, Prevención, Pudor, Pusilanimidad, Reserva, Resquemor, Sometimiento, Sumisión, Suspicacia, Susto, Temor, Terror, Timidez, Vergüenza, Victimismo, Vigilancia, Vulnerabilidad'
sorpresa = 'Alteración, Ambivalencia, Anormalidad, Arrobamiento, Asombro, Aturdimiento, Confusión, Conmoción, Curiosidad, Desapercibimiento, Desconcierto, Deslumbramiento, Desorientación, Disonancia, Distracción, Duda, Emergencia, Escepticismo, Estremecimiento, Estupefacción, Estupor, Expectación, Extrañeza, Extraordinario, Extravagancia, Ilógico, Impacto, Impresión, Imprevisión, Inadvertencia, Inaudito, Incoherencia, Incongruencia, Incredulidad, Increíble, Indecisión, Inesperado, Insólito, Insospechado, Intempestivo, Intriga, Irreal, Irreflexión, Maravilla, Obnubilación, Pasmo, Peculiaridad, Perplejidad, Prodigio, Rareza, Revelador, Shock, Sobresalto, Traumático, Titubeo, Urgencia, Vacilación'
asco = 'Abominable, Aborrecimiento, Abstinencia, Alejamiento, Animadversión, Antagonismo, Antipatía, Apartamiento, Aversión, Censura, Contención, Continencia, Desafecto, Desapego, Desagrado, Desavenencia, Desazón, Desdeño, Desprecio, Destitución, Discrepancia, Discriminación, Disentimiento, Distanciamiento, Divergencia, Eliminación, Emético, Empacho, Empalago, Evasión, Evitación, Execración, Grima, Hastío, Inadecuación, Inapetencia, Incompatibilidad, Inadmisión, Inmoralidad, Inmundicia, Inquina, Intolerancia, Mesura, Moderación, Náusea, Nocividad, Obscenidad, Oposición, Rechazo, Remoción, Renuencia, Renuncia, Repudio, Repugnancia, Repulsión, Separación, Sobriedad, Tirria, Toxicidad, Vomitivo'


tristeza = tristeza.split(',')
alegria = alegria.split(',')
enfado = enfado.split(',')
amor = amor.split(',')
miedo = miedo.split(',')
sorpresa = sorpresa.split(',')
asco = asco.split(',')


sentimientos = tristeza + alegria + enfado + amor + miedo + sorpresa + asco
sentimientos_lemma = []

existential_tot = ['haber', 'pararse',  'elevarse', 'quedarse',  'emerger', 'suceder', 'producirse', 'presentarse', 'existir',  'seguir', 'predominar',  'resultar', 'aparecer', 'florecer', 'permanecer', 'estar',  'radicar', 'ascender', 'quedar', 'prosperar',  'imperar', 'surgir',  'sobrevivir',  'ocurrir', 'consistir', 'aflorar', 'madurar',  'colocar',  'acostarse',  'levantarse', 'echarse',  'sostenerse', 'yacer', 'alzarse', 'crecer',   'encontrarse',  'prevalecer', 'tenderse',  'salir', 'vivir',  'extenderse']
behavioural_tot = ['cantar', 'llorar', 'sonreír',  'gemir', 'eructar', 'sollozar', 'chillar', 'estornudar',  'quejarse', 'toser', 'vigilar',  'sentarse',  'parlotear', 'mirar', 'lloriquear', 'aspirar',  'cabecear',  'charlar',   'suspirar', 'abuchear',  'respirar', 'silbar', 'reír', 'transpirar',  'hipar', 'dormir',  'susurrar', 'gimotear', 'cagar',  'tumbarse',   'posar',  'entonar',  'sisear',  'gritar',  'observar',  'refunfuñar',  'bufar']
relational_poss = ['considerar','poseer', 'merecer','tener' ,'necesitar', 'pertenecer','incluir', 'involucrar','excluir', 'apoderarse de']
relational_circ = ['saludar', 'preceder','durar', 'exceder', 'superar','faltar','prevenir','parecerse', 'rodear', 'contradecir', 'abarcar', 'llegar', 'cubrir', 'causar','concordar','encajar']
relational_inte = ['buscar','ser', 'costar', 'medir' , 'ilustrar', 'significar', 'asegurar','garantizar', 'confirmar','representar','constituir','ejemplificar','sumar','igualar','formar','diferir', 'aventajar a',   'recorrer',  'ir', 'variar','doler','elegir','escoger']
material_cret = ['provocar', 'destapar',  'sacar', 'jalar', 'tejer',  'erigir',  'coser',  'esbozar',  'contraer',  'instituir',  'ganar',  'fundar',  'generar',  'tomar', 'avanzar',  'encontrar', 'perforar', 'estructurar', 'adquirir', 'remover', 'entrenar',  'desenfundar',  'entablar', 'dirigirse', 'escarbar', 'investigar', 'empatar', 'levantar',  'guisar', 'practicar', 'aumentar', 'tramar', 'presentar', 'construir', 'serpentear', 'atraer', 'edificar', 'nombrar', 'establecer',  'preparar', 'revelar', 'zigzaguear', 'entrenar', 'desarrollar', 'taladrar', 'mostrar',  'falsificar', 'hacer', 'forjar', 'delinear', 'redactar', 'intercalar',  'configurar', 'afirmar', 'abrir', 'meter', 'consagrar', 'evolucionar', 'crear', 'cobrar', 'efectuar', 'montar', 'instruir',  'cometer', 'excavar', 'trazar', 'tricotar', 'cocinar', 'fraguar', 'abrirse', 'diseñar',  'separar', 'cavar', 'elaborar', 'desenvainar', 'producir', 'armar', 'amañar',  'urdir', 'explotar',  'dibujar', 'escribir', 'entrelazar', 'poner', 'dar','seleccionar', 'determinar', 'arrancar', 'elaborar',  'urbanizar', 'reclutar', 'componer', 'agujerear', 'planear', 'comenzar', 'perfeccionar', 'iniciar', 'fabricar']
material_tras =  ['respetar', 'permitir', 'velar','resguardar','proteger','cuidar','mantener','realizar'    'tejer', 'conferir', 'resbalar', 'regular', 'perjudicar',  'deshollinar', 'abandonar', 'eliminar',  'marchitar', 'legar', 'chasquear', 'allanar', 'recaudar', 'irradiar', 'dispersar', 'perforar', 'abonar', 'aderezar', 'fusilar',   'dominar',  'desnudar', 'obedecer', 'descender', 'descarozar', 'consolidarse', 'clavar', 'levantar', 'reunirse', 'licuar', 'acallar', 'bañar', 'retirarse', 'desinvertir', 'hackear', 'tronar', 'envejecer', 'aumentar', 'regresar', 'curvar', 'fugarse', 'entregar', 'joderse', 'acabar', 'estirar', 'agitar', 'vestir', 'violar', 'barnizar', 'unir', 'alquitranar', 'atravesar', 'conmover', 'adelantar', 'preparar', 'estrellar', 'impresionar',  'titilar', 'abofetear', 'expandirse', 'alumbrar', 'desvainar', 'enfrentar', 'estipular', 'cortar', 'caerse', 'funcionar',  'parar', 'curtir', 'distribuir', 'oscilar', 'telegrafiar', 'arañarse', 'prestar', 'restar', 'volcarse', 'amputar', 'sisar', 'enyesar', 'desplegado', 'acunar', 'techar', 'inyectarse', 'actuar', 'pautar', 'resquebrajar', 'pulverizar', 'borrar', 'transcurrir', 'pasar', 'obrar', 'decaer', 'domar', 'chirriar', 'cobrar',
 'adoquinar', 'concluir', 'dispersarse', 'reducir', 'recoger', 'desempolvar', 'tirar', 'inclinar', 'rebotar', 'remendar', 'rendir',  'presionar', 'incitar', 'desportillar', 'debilitar', 'otorgar', 'derrocar', 'rebanar', 'repetirse', 'comer', 'balancear', 'calentarse', 'producir', 'gobernar', 'rizar',  'venderse', 'formular', 'dispararse',  'tapizar', 'llenar', 'arrendar', 'comprar',  'aprobar', 'moler', 'brillar', 'reafirmar', 'sustituir',  'determinar', 'anotar', 'usar', 'arrancar', 'destrozar', 'filmar', 'venirse', 'dañar',  'ribetear', 'azotar', 'segar', 'estremecer', 'destituir',  'contorsionar', 'desenvolver', 'retar', 'arrollar',  'perder',  'sacar', 'librar', 'prolongarse', 'plegar', 'revolver', 'descomponer', 'impulsar', 'privar', 'alterar', 'actualizar', 'arrugar',  'rozar', 'desmentir', 'moldear', 'conceder', 'capitanear', 'pescar',  'fundir', 'decorar', 'agrupar', 'avanzar',  'disminuir', 'patalear', 'alquilar', 'arrasar', 'agrandar', 'simplificar', 'anillar', 'entablar',  'suplir', 'recibir',  'adoptar', 'centrifugar', 'templar', 'cruzar', 'aplastar', 'estimular', 'aterrizar', 'modernizar', 'despedir', 'adelgazar', 'vencer', 'pasear', 'registrar',
 'pronosticar', 'reparar', 'curar',  'tallar', 'tergiversar', 'amueblar', 'propagar', 'deslizar', 'regalar', 'renguear', 'mostrar', 'sanar', 'bramar', 'enlucir', 'ensombrecer', 'reconocer', 'hacer', 'hurtar', 'colarse', 'remorder', 'regir', 'diluir', 'descontar', 'afectar', 'quemar', 'regañar', 'repiquetear', 'reventar', 'rasquetear', 'partir', 'enfocar', 'admitir', 'ennegrecer', 'hundir', 'señalar', 'atestar', 'distinguir',   'rasgar', 'desarmar', 'continuar', 'licenciar', 'recopilar', 'doblar', 'exagerar', 'destacar', 'amortiguar', 'cabalgar', 'remachar', 'columpiarse', 'montar', 'amamantar', 'apuñalar', 'labrar', 'atenuar',  'potenciar', 'desmontar', 'asaltar', 'infringir', 'rasurar', 'rifar', 'encubrir', 'desvanecer',  'zarandear', 'vaciar','devolver', 'estirarse', 'dirigir', 'piratear', 'concentrar', 'deshacer', 'elaborar', 'corregir', 'proporcionar', 'armar',  'publicar', 'reinar', 'talar', 'romper', 'postularse', 'deshuesar', 'robar', 'aumentarse', 'fustigar', 'volcar', 'poner', 'desembarcar', 'dar', 'descargar',
 'retirar', 'resonar', 'llevarse', 'afianzar', 'telefonear', 'refregar', 'plantear', 'expulsar', 'freír', 'quitar', 'separado', 'rebañar', 'subir', 'derretir', 'incendiar', 'prolongar', 'disparar', 'inflar', 'perfeccionar', 'demorarse', 'matar', 'añejar', 'proveer', 'descascarillar', 'espetar',  'exponer', 'intervenir', 'destapar', 'divulgar', 'chapar', 'ampliar', 'enrollar', 'dividir', 'juntar', 'lubricar', 'disipar', 'desvelar', 'estrujar', 'clausurar',  'atropellar', 'aliviar', 'accionar',  'acuñar', 'prender', 'envolver', 'equilibrar', 'prensar', 'apagar', 'retorcer', 'calar',  'revolcarse', 'aguantar', 'apiñarse', 'pinchar', 'adquirir', 'endurecer',  'encerar', 'cortarse', 'despellejar', 'frotar', 'sacrificar', 'desprenderse', 'alinear', 'aplicar', 'forrar', 'patear', 'tañer', 'improvisar',  'incorporar', 'incrementar', 'engrasar',  'presentar', 'palidecer', 'serpentear', 'coleccionar', 'atraer', 'encoger', 'aprovechar', 'enrojecer', 'ceder', 'desaparecer', 'friccionar', 'resolver', 'volver', 'trastornar', 'contraerse', 'adornar', 'enviar',  'blindar', 'interceptar', 'resplandecer', 'esparcir', 'sembrar', 'rebozar', 'incrementarse', 'cerrar', 'reunir',  'manchar', 'suprimir', 'manejar', 'trocear', 'tender',  'anticipar', 'galopar', 'arrear',  'torcerse', 'sobrehilar',  'arder',  'capturar', 'satisfacer', 'dilatar', 'abrir', 'cumplir', 'freírse', 'batir',
 'desplegar', 'mitigar',  'volarse', 'expandir', 'ensartar', 'mancillar', 'chocar', 'demoler', 'extirpar', 'lapidar', 'empujar', 'quebrantar', 'juntarse', 'desbullar', 'volar', 'esclarecer', 'cachar', 'obtener', 'rematar', 'colorear', 'encrespar', 'faxear', 'exigir', 'pintar', 'pulsar', 'consolidar', 'tardar', 'descruzar', 'aceitar', 'naufragar', 'rajar', 'refrigerar', 'brindar', 'acumularse', 'desteñir', 'ratear', 'agrietar',  'apedrear', 'repartir', 'desgarrar', 'agarrar', 'explotar', 'sobresaltarse', 'conducir', 'arruinar', 'destruir', 'suministrar', 'estropear', 'denegar', 'confluir', 'sobornar', 'recortar', 'iluminar', 'virar',  'golpear',  'achicarse',  'caer', 'cascarse', 'acuchillar', 'cepillarse', 'afeitar', 'empedrar', 'agujerear', 'desdoblar','repicar', 'fomentar', 'pelear', 'quebrar', 'herir', 'vendar', 'allanarse', 'sobrevenir', 'limpiar', 'revender', 'despojar', 'rodar', 'enfriar', 'desviarse', 'abrasar',  'traer', 'temblar', 'esmaltar', 'ensamblar', 'abundar', 'saltar', 'abarrotar', 'lanzar', 'extender', 'enrular', 'despegar', 'refrescar', 'omitir', 'pulir', 'reverberar', 'atender', 'fracasar', 'mejorar',  'degradar', 'hinchar', 'arreglar', 'alcanzar', 'redoblar', 'apuntar',  'restregar', 'desparramar', 'enlosar', 'imprimir', 'tomar', 'consumir', 'soltar', 'deformar', 'progresar', 'facilitar',  'difundir', 'estremecerse', 'explosionar',  'tumbar', 'ladrar',
 'coger', 'introducir', 'colapsar','cambiar', 'masajear', 'esquivar', 'cocer', 'ataviar', 'entrar', 'refulgir', 'fallar', 'cepillar', 'destripar', 'grabar',  'pavimentar', 'alargar', 'aprovisionar', 'cicatrizar',  'tachar', 'restallar', 'acumular', 'dejar',  'descomprimir', 'aclararse', 'disolver', 'donar', 'ejecutar', 'revivir', 'dictaminar', 'nombrar', 'cachetear', 'aplanar', 'derribar', 'revelar', 'flaquear', 'cazar', 'torcer', 'lustrar', 'ladear', 'blanquear', 'importar', 'retumbar', 'amortajar', 'reforzar', 'cancelar', 'secar', 'marchar', 'mecer', 'terminar', 'engañar', 'disimular', 'obsequiar', 'relucir',  'suavizar', 'soportar',  'tratar', 'desvestir', 'bastar', 'disolverse', 'ofrecerse', 'ofrecer',  'controlar', 'criar', 'agotar', 'encender', 'condensar',  'desconectar', 'podar', 'operar', 'zurcir', 'evitar',  'tamborilear', 'distorsionar', 'apurado', 'lacio', 'mezclar', 'imitar', 'anular', 'golpetear', 'ensanchar', 'saltarse', 'marcar', 'enladrillar', 'dilatarse', 'triturar', 'dorar', 'curarse',  'inscribirse','separar',  'rebasar',  'arañar',
 'arriar', 'echar', 'pegar', 'acrecentar', 'conseguir', 'hilar', 'destellar', 'unirse', 'copiar', 'contestar', 'difundirse', 'quitarse', 'calentar', 'ablandar', 'colgar', 'tapar',  'revocar', 'estallar', 'sangrar', 'estafar', 'pilotear', 'deducirse', 'arponear', 'batear', 'disiparse',  'atacar', 'apisonar',  'amainar', 'descubrir', 'promover',  'rentar', 'oscurecer', 'adjudicar', 'miniar', 'contratar', 'timar', 'llevar',  'cundir', 'servir', 'rebajar', 'extraer', 'trabajar',  'orientar', 'animar', 'exprimir', 'comprimir', 'bajar', 'comenzar', 'fortalecer',  'oponer', 'iniciar', 'alimentar', 'vender']
material_tot = material_cret + material_tras
relational_tot = relational_poss + relational_inte

mental_perc = ['percibir','ver','oír','notar','escuchar','detectar',  'oler', 'sentir','palpar','degustar','entrever','fijarse','tocar', 'vislumbrar']
mental_cogn = ['valorar','entender','saber','comprender','suponer','pensar','percatar','preguntar','conocer','hipotetizar','soñar','recordar', 'prever','dudar','conjeturar','imaginar','olvidar','creer','adivinar','calcular','reflexionar' ] 
mental_desi = ['desear','querer','gustar','coincidir','rebelarse','rechazar','rehusar','complacer','negarse','acatar', 'decidir','consentir','acceder','aceptar', 'añorar']
mental_emot = ['sufrir','alegrar','lamentar','arrepentir','odiar','molestar','preocupar','tener miedo','temer','afligir', 'amar', 'horrorizar', 'disgustar','gozar','encantar',  'detestar','hartar','aburrir', 'escandalizar', 'deprimir', 'tranquilizar','repugnar']
mental = mental_perc + mental_cogn + mental_desi + mental_emot

parte_del_cuerpo = ['pecho','corazón','estomago','cabeza','cuello','ojos','boca','cara','rostro','espalda','piernas']


halliday_dict = {
    'EXISTENTIAL': existential_tot,
    'BEHAVIOURAL': behavioural_tot,
    'RELATIONAL': relational_tot,
    'MATERIAL': material_tot,
    'SENTIMENT':sentimientos_lemma,
    'MENTAL': mental,
    'BODY_PART': parte_del_cuerpo
}


nlp = stanza.Pipeline(lang= 'es', dir= nb_path, processors='tokenize,mwt,pos,lemma,depparse')

data = pd.read_csv('../files/data_preprocesada.csv',
                                     sep = ";",
                                     usecols = [0,8]
                                     )

sintactico = {
    'id':[],
    'PRON_MENTAL_SENTIMENT':[],
    'PRON_BEHAVIOURAL':[],
    'PRON_BODY_PART_BEHAVIOURAL':[],
    'NO_MENTAL':[]
}

for index, row in data.iterrows():
    print(index)
    current_row = row['text2']
    phrase_list = re.split(';',current_row)
    sintactico['id'].append(row['Identifier'])

    suma_pron_mental_sentiment = 0
    suma_pron_BEHAVIOURAL = 0
    suma_pron_BODY_PART_BEHAVIOURAL = 0
    suma_no_mental = 0


    for p in phrase_list:
        pattern = ['PRON','MENTAL', 'SENTIMENT']
        pron_mental_sentiment = count_patterns_found(p, pattern, nlp)
        suma_pron_mental_sentiment = suma_pron_mental_sentiment + pron_mental_sentiment

        pattern = ['PRON','BEHAVIOURAL']
        pron_BEHAVIOURAL = count_patterns_found(p, pattern, nlp)
        suma_pron_BEHAVIOURAL = suma_pron_BEHAVIOURAL + pron_BEHAVIOURAL

        pattern = ['PRON','BODY_PART','BEHAVIOURAL']
        pron_BODY_PART_BEHAVIOURAL = count_patterns_found(p, pattern, nlp)
        suma_pron_BODY_PART_BEHAVIOURAL = suma_pron_BODY_PART_BEHAVIOURAL + pron_BODY_PART_BEHAVIOURAL        
    
        pattern = ['no','MENTAL']
        no_mental = count_patterns_found(p, pattern, nlp)
        suma_no_mental = suma_no_mental + no_mental


    sintactico['PRON_MENTAL_SENTIMENT'].append(suma_pron_mental_sentiment)
    sintactico['PRON_BEHAVIOURAL'].append(suma_pron_BEHAVIOURAL)
    sintactico['PRON_BODY_PART_BEHAVIOURAL'].append(suma_pron_BODY_PART_BEHAVIOURAL)
    sintactico['NO_MENTAL'].append(suma_no_mental)


marcadores_sintacticos = pd.DataFrame(sintactico)
marcadores_sintacticos.to_csv('../files/marcadores_sintacticos.csv', index= False)