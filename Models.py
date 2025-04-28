from Datos_BBDD import db # importamos el contenido de la base de datos para que podamos acceder a el

''' Creamos una clase llamada steamgamesdb
Esta clase va a ser nuestro modelo de datos de los juegos (el cual nos servirá luego para la base de datos)'''


class steamgamesdb(db.Model):
    __tablename__ = "steamgamesdb"
    title = db.Column(db.String(200), primary_key=True)
    recommendations = db.Column(db.String())

    def repr(self):  # ? Copia de __str__
        return "{}: {}".format(self.title, self.recommendations)

    def __str__(self):
        return "{}: {}".format(self.title, self.recommendations)

