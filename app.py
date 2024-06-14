from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import base64
from io import BytesIO
import time

app = Flask(__name__)

@app.route('/')
def variables():
    return render_template('input.html')


@app.route('/output', methods=['POST'])
def calculate():
    if request.method == 'POST':
        # Obtenir les données du formulaire
        D_AB_exp = float(request.form['D_AB_exp'])  # cm^2/s
        T    = float(request.form['T']) # K
        Xa   = float(request.form['Xa'])
        Xb   = float(request.form['Xb'])
        r_a  = (float(request.form['r_a']))**(1/3)# r_a = ra^(1/3)
        r_b  = (float(request.form['r_b']))**(1/3) # r_b = rb^(1/3)
        q_a  = float(request.form['q_a'])
        q_b  = float(request.form['q_b'])
        D0AB = float(request.form['D0AB'])
        D0BA = float(request.form['D0BA'])

        # Fonction pour calculer D_AB
        def calculate_D_AB(params):
            a12, a21 = params
            # Calcul des termes de l'équation
            term1 = Xb*np.log(D0AB) + Xa*D0BA + 2*(Xa*np.log((Xa*r_a+Xb*r_b)/r_a)+Xb*np.log((Xb*r_b+Xa*r_a)/r_b))
            term2 = 2*Xa*Xb*((r_a/(Xa*r_a+Xb*r_b))*(1-(r_a/r_b)) + (r_b/(Xa*r_a+Xb*r_b))*(1-(r_b/r_a)))
            term3 = Xb*q_a*((1-((Xa*q_a*np.exp(-a21/T))/(Xa*q_a+Xb*q_b*np.exp(-a21/T)))**2)*(-a21/T)+(1-((Xb*q_b)/(Xb*q_b+Xa*q_a*np.exp(-a12/T)))**2)*np.exp(-a12/T)*(-a12/T))
            term4 = Xa*q_b*((1-((Xb*q_b*np.exp(-a12/T))/(Xa*q_a*np.exp(-a12/T)+Xb*q_b))**2)*(-a12/T)+(1-((Xa*q_a)/(Xa*q_a+Xb*q_b*np.exp(-a21/T)))**2)*np.exp(-a21/T)*(-a21/T))
            # Calcul de D_AB
            D_AB = np.exp(term1 + term2 + term3 + term4)
            return D_AB

        # Fonction objectif pour la minimisation
        def objective(params):
            D_AB_calculated = calculate_D_AB(params)
            return (D_AB_calculated - D_AB_exp)**2

        # Paramètres initiaux
        params_initial = [1.0, 1.0]

        # Erreur initiale
        error = float('inf')

        # Tolerance
        tolerance = 1e-8

        # Nombre maximal d'itérations
        max_iterations = 1000
        iteration = 0

        # Boucle d'ajustement des paramètres
        while error > tolerance and iteration < max_iterations:
            # Minimisation de l'erreur
            result = minimize(objective, params_initial, method='Nelder-Mead')
            # Paramètres optimisés
            a12_opt, a21_opt = result.x
            # Calcul de D_AB avec les paramètres optimisés
            D_AB_opt = calculate_D_AB([a12_opt, a21_opt])
            # Calcul de l'erreur
            error = abs(D_AB_opt - D_AB_exp)
            # Mise à jour des paramètres initiaux
            params_initial = [a12_opt, a21_opt]
            # Incrémentation du nombre d'itérations
            iteration += 1

        # Affichage des résultats
        print("Paramètres optimisés:")
        print("a12 =", a12_opt)
        print("a21 =", a21_opt)
        print("D_AB calculé avec les paramètres optimisés:", D_AB_opt)
        print("erreur =", error)

        Xa_values = np.linspace(0, 0.7, 100)  # Fraction molaire de A

        debut = time.time()
        # Fonction pour calculer D_AB à partir des coefficients a_AB et a_BA
        def calculate_DAB(Xa):
            a12 = 236.82636803699774
            a21 = 1268.1042421722684
            Xb = 1 - Xa  # Fraction molaire de B
            
            # Calcul des termes de l'équation
            term1 = Xb*np.log(D0AB) + Xa*D0BA + 2*(Xa*np.log((Xa*r_a+Xb*r_b)/r_a)+Xb*np.log((Xb*r_b+Xa*r_a)/r_b))
            term2 = 2*Xa*Xb*((r_a/(Xa*r_a+Xb*r_b))*(1-(r_a/r_b)) + (r_b/(Xa*r_a+Xb*r_b))*(1-(r_b/r_a)))
            term3 = Xb*q_a*((1-((Xa*q_a*np.exp(-a21/T))/(Xa*q_a+Xb*q_b*np.exp(-a21/T)))**2)*(-a21/T)+(1-((Xb*q_b)/(Xb*q_b+Xa*q_a*np.exp(-a12/T)))**2)*np.exp(-a12/T)*(-a12/T))
            term4 = Xa*q_b*((1-((Xb*q_b*np.exp(-a12/T))/(Xa*q_a*np.exp(-a12/T)+Xb*q_b))**2)*(-a12/T)+(1-((Xa*q_a)/(Xa*q_a+Xb*q_b*np.exp(-a21/T)))**2)*np.exp(-a21/T)*(-a21/T))
            # Calcul de D_AB
            D_AB = np.exp(term1 + term2 + term3 + term4)
            return D_AB

        # Calcul de D_AB avec les paramètres optimisés
        D_AB_values = [calculate_DAB(Xa) for Xa in Xa_values]

        fin = time.time()
        temps = fin-debut 

        # Tracer la variation du coefficient de diffusion en fonction de la fraction molaire
        plt.plot(Xa_values, D_AB_values)
        plt.xlabel('Fraction molaire de A')
        plt.ylabel('Coefficient de diffusion (cm^2/s)')
        plt.title('Variation du coefficient de diffusion en fonction de la fraction molaire')
        plt.grid(True)

        # Encodage de l'image en base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Attendre 3 secondes avant de rediriger vers la page d'accueil
        time.sleep(3)

        # Renvoyer le modèle output.html avec les résultats calculés
    return render_template('output.html', a12=a12_opt, a21=a21_opt, error=error, D_AB=D_AB_opt, plot_encoded=plot_encoded,iteration=iteration,temps=temps)

