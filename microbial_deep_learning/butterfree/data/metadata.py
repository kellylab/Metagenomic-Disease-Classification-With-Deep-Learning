import butterfree
import pandas as pd
import os
import numpy as np
import functools
from itertools import count
from collections import defaultdict
import pickle
from fireworks.extensions import database as db
from sqlalchemy import create_engine, Column, Integer, String, Float

def get_cmd_metadata():

    conn_string = "postgresql://{user}:{pwd}@{host}:{port}/{db}".format(**{
        'user': butterfree.username, 'pwd': butterfree.password,
        'host': butterfree.host, 'port': butterfree.port,
        'db': butterfree.dbname
    })
    engine = create_engine(conn_string)
    metadata = db.DBPipe('phenotypes', engine)

    return metadata

def write_cmd_metadata(path):
    """
    Downloads metadata from CMD and writes it to a CSV file at chosen path.
    """

    metadata = get_cmd_metadata()
    message = metadata.all()
    df = message.df
    df.to_csv(path)
