SETUP
=====

Assume root directory is `/home/ubuntu/mflux`.

Python environment
------------------
1. `sudo apt-get install python-dev`
2. `sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose`
3. `sudo pip install scikit-learn python-constraint`
4. regenerate all models by running `python get_model.py`

web server
----------
- `apt-get install nginx uwsgi`

nginx
-----

sudo vim /etc/nginx/sites-enabled/default

```
server {
        listen 80 default_server;
        listen [::]:80 default_server ipv6only=on;

        root /home/ubuntu/mflux;
        index index.html index.htm;

        server_name mflux.org;
        location / {
                index index.html;
        }

        location ~ \.py$ {
                include uwsgi_params;
                uwsgi_modifier1 9;
                uwsgi_pass 127.0.0.1:9000;
        }
}
```

uwsgi
-----

sudo vim /etc/uwsgi/apps-enabled/mflux.ini

```
[uwsgi]
plugins = cgi
socket = 127.0.0.1:9000
chdir = /home/ubuntu/mflux/
module = pyindex
cgi=/=/home/ubuntu/mflux/
cgi-helper =.py=python

```

`sudo service restart uwsgi`
`sudo service restart nginx`
