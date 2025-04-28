import ast
from flask import Flask, render_template, request, jsonify
from Models import steamgamesdb as sg
from Datos_BBDD import db

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://Admin:123456@localhost/SteamgamesDB'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Evitar advertencias innecesarias

# Inicializamos la base de datos con la aplicación
db.init_app(app)

# Esta sera nuestra portada de pagina
@app.route('/')
def index():
    game_title = request.args.get('gameTitle', '').strip()  # Obtiene el término de búsqueda del formulario
    page = request.args.get('page', 1, type=int)  # Página actual para la paginación

    # Si se proporciona un término de búsqueda
    if game_title:
        juegos = sg.query.filter(sg.title.like(f"%{game_title}%")).paginate(page=page, per_page=10, error_out=False)
    else:
        # Si no se proporciona ningún término de búsqueda, mostramos los primeros 10 juegos
        juegos = sg.query.paginate(page=page, per_page=10, error_out=False)

    # Convertimos las recomendaciones a una cadena separada por comas (si son listas)
    for juego in juegos.items:
        # Primero verificamos si la recomendación es una cadena en el formato de lista
        try:
            # Intentamos convertir la cadena con las comillas extrañas a una lista real
            recomendaciones = ast.literal_eval(juego.recommendations)
            if isinstance(recomendaciones, list):  # Si se convierte en lista correctamente
                juego.recommendations = ', '.join(recomendaciones)  # Convertir la lista a cadena separada por comas
        except (ValueError, SyntaxError):
            # Si no se puede convertir, dejamos la recomendación tal cual está
            pass

    return render_template('index.html', lista_juegos=juegos.items, pagination=juegos)

# Ruta para obtener sugerencias en tiempo real mientras se escribe
@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('q', '').strip()  # Obtiene la consulta de búsqueda
    suggestions = []
    if query:
        # Buscar juegos que comienzan con la cadena de búsqueda
        suggestions = sg.query.filter(sg.title.like(f"{query}%")).limit(10).all()
        suggestions = [juego.title for juego in suggestions]  # Solo devolver los títulos
    return jsonify(suggestions=suggestions)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Crear las tablas si no existen
    app.run(debug=True)
