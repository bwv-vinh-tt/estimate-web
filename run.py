import os
import platform
from sys import exit

from flask import render_template
from flask_migrate import Migrate
from flask_minify import Minify

from apps import create_app, db
from apps.config import config_dict

# check platform
print(platform.machine())
print(platform.architecture())
print(platform.platform())

# WARNING: Don't run with debug turned on in production!
DEBUG = os.getenv("DEBUG", "False") == "True"

# The configuration
get_config_mode = "Debug" if DEBUG else "Production"

try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit("Error: Invalid <config_mode>. Expected values [Debug, Production] ")

app = create_app(app_config)

# Errors


@app.errorhandler(403)
def access_forbidden(error):
    return render_template("home/page-403.html"), 403


@app.errorhandler(404)
def not_found_error(error):
    return render_template("error/404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("home/page-500.html", error=error), 500


Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

if DEBUG:
    app.logger.info("DEBUG            = " + str(DEBUG))
    app.logger.info("Page Compression = " + "FALSE" if DEBUG else "TRUE")
    app.logger.info("DBMS             = " + app_config.SQLALCHEMY_DATABASE_URI)

if __name__ == "__main__":
    app.run(port=5000)
