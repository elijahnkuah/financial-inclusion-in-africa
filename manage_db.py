# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 03:41:37 2021

@author: Elijah_Nkuah
"""
import sqlite3
#conn = sqlite3.connect("usersdata.db")
#c = conn.cursor()


# FXN
def create_usertable():
    with sqlite3.connect("usersdata.db") as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    #conn.close()
def add_userdata(username, password):
    with sqlite3.connect("usersdata.db") as conn:
        conn.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username,password))
        conn.commit()
    #conn.close()
def login_user(username, password):
    with sqlite3.connect("usersdata.db") as conn:
        data = list(
            conn.execute('SELECT * FROM userstable WHERE username =? AND password =?',
                     (username,password)))
    return data
    #conn.close()
    
def view_all_users():
    with sqlite3.connect("usersdata.db") as conn:
        conn.execute('SELECT * FROM userstable')
        data = conn.fetchall()
    return data
    
#conn.close()