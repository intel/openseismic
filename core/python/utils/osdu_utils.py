#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
from json import dumps, loads
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import base64


def get_access_token(url, client_id, client_secret, refresh_token):
    secret = bytes(u'%s:%s' % (client_id, client_secret), 'utf-8')
    encoded_secret = base64.b64encode(secret).decode('utf-8')
    body = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': client_id,
        'client_secret': client_secret
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': 'Basic %s' % (encoded_secret)
    }
    data = urlencode(body).encode('utf8')
    request = Request(url=url, data=data, headers=headers)
    response_body = {}

    try:
        response = urlopen(request)
        response_b = response.read()
        response_body = loads(response_b.decode("utf-8"))

        id_token = response_body['id_token']
        access_token = response_body['access_token']
        expires_in = response_body['expires_in']
        token_type = response_body['token_type']

        return id_token, access_token, expires_in, token_type
    except HTTPError as e:
        code = e.code
        message = e.read().decode('utf8')
        print(f"Refresh token request failed. {code} {message}")


def get_sample_data(access_token):
    search_url = 'https://3hfmgxokqc.execute-api.us-east-1.amazonaws.com'
    search_url = search_url + '/api/search/v2/query'
    payload = str(dumps({
        "kind": "opendes:osdu:seismictracedata-wpc:0.2.0",
        "limit": 1, "aggregateBy": "kind"
    }))
    headers = {
        'Content-Type': 'application/json',
        'data-partition-id': 'opendes',
        'Authorization': 'Bearer %s' % (access_token)
    }
    data = payload.encode('utf8')
    request = Request(url=search_url, data=data, headers=headers)

    try:
        response = urlopen(request)
        response_b = response.read()
        response_body = loads(response_b.decode("utf-8"))

        return response_body
    except HTTPError as e:
        code = e.code
        message = e.read().decode('utf8')
        print(f"Refresh token request failed. {code} {message}")


def get_file_from_srn(srn, access_token):
    # Expects srn to be an array
    delivery_url = 'https://3hfmgxokqc.execute-api.us-east-1.amazonaws.com'
    delivery_url = delivery_url + '/api/file/v1/GetFileSignedUrl'
    payload = str(dumps({"srns": srn}))
    headers = {
        'Content-Type': 'application/json',
        'data-partition-id': 'opendes',
        'Authorization': 'Bearer %s' % (access_token)
    }
    data = payload.encode('utf8')
    request = Request(url=delivery_url, data=data, headers=headers)

    try:
        response = urlopen(request)
        response_b = response.read()
        response_body = loads(response_b.decode("utf-8"))

        return response_body
    except HTTPError as e:
        code = e.code
        message = e.read().decode('utf8')
        print(f"Refresh token request failed. {code} {message}")
