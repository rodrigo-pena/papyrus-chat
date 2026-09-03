# Share a local Papyrus Chat demo

This guide lets a collaborator use Papyrus Chat from their own browser while
the application, corpus, database, model endpoint, and credentials remain on
your computer. Papyrus Chat stays bound to localhost; ngrok provides the
temporary public HTTPS URL.

## Before you start

You need:

- A working Papyrus Chat checkout and corpus artifact.
- An OpenAI-compatible model endpoint configured with `LLM_BASE_URL`,
  `LLM_MODEL`, and, when required, `LLM_API_KEY`.
- [ngrok installed and authenticated](https://ngrok.com/download) on your
  computer.
- A long, random password to share with the collaborator through a separate,
  trusted channel.

Create `conf/ngrok-policy.yml` locally in the repository checkout. The path is
intentionally listed in `.gitignore` because the file contains the Basic Auth
credential. Replace the password placeholder before starting ngrok; recreate
the file from the example below if it is missing.

## 1. Start Papyrus Chat locally

From the repository checkout, start the app in one terminal:

```bash
uv run papyrus-chat --artifact ./data/papyrus-corpus \
  --host 127.0.0.1 --port 8000 --no-open
```

Keeping the bind address at `127.0.0.1` means the application is not directly
listening on the local network. Leave this terminal running.

## 2. Configure the ngrok Traffic Policy

Create `conf/ngrok-policy.yml` and replace the placeholder with the password
you will send to the collaborator:

```yaml
on_http_request:
    - actions:
          - type: basic-auth
            config:
                realm: "papyrus-chat"
                credentials:
                    - "collaborator:REPLACE_WITH_A_LONG_RANDOM_PASSWORD"
                enforce: true

          # Pydantic AI validates Host headers and accepts this local host by default.
          - type: add-headers
            config:
                headers:
                    host: "127.0.0.1"
```

The Basic Auth action protects the public endpoint. The header rewrite avoids
Pydantic AI's hostname validation error for the generated ngrok hostname. In
this repository, the locked `pydantic-ai-slim` version enables that validation,
while the application does not currently configure `allowed_hosts`.

## 3. Start the tunnel

In a second terminal, run ngrok and point it at port 8000:

```bash
ngrok http 8000 --traffic-policy-file ./conf/ngrok-policy.yml
```

The terminal displays a temporary HTTPS forwarding URL similar to:

```text
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

Send the `https://...ngrok-free.app` URL and the username/password to the
collaborator through a trusted channel. They open the URL in Chrome, Edge, or
Firefox; they do not need Python, this repository, the corpus, or ngrok.

Their browser first shows the Basic Auth prompt. Use:

```text
Username: collaborator
Password: the password from ngrok-policy.yml
```

The request path is:

```text
Collaborator's browser
        |
        | HTTPS + Basic Auth
        v
      ngrok
        |
        | http://127.0.0.1:8000
        v
Papyrus Chat on your computer
        |
        +-- local corpus and SQLite database
        +-- local or remote LLM endpoint
        +-- server-side credentials
```

## Stop sharing

Press `Ctrl+C` in the ngrok terminal and in the Papyrus Chat terminal when the
demo is finished. The temporary URL stops working when either process stops,
your computer loses internet access, or your computer is shut down.

## Security notes

- Share the URL only with a trusted collaborator. The tunnel makes the chat UI
  reachable from the public internet.
- Keep `conf/ngrok-policy.yml` and its password private. Rotate the password
  after the demo, especially if it was sent through a channel you do not
  control.
- Anyone who passes Basic Auth can submit prompts through your running app and
  cause it to use its configured model endpoint and local corpus. Use a model
  provider and corpus appropriate for this level of access.
- The browser does not receive your local API key directly, but the Papyrus
  Chat process can access whatever local files and services its user account
  can access.
- Do not use an unprotected `ngrok http 8000` command for a shared demo.

If the collaborator receives `421 Misdirected Request`, check that ngrok was
started with this policy file and that both the Basic Auth and `add-headers`
actions are present. A normal tunnel hostname is rejected by the application's
current Host validation unless it is rewritten to `127.0.0.1`.

See ngrok's [localhost sharing quickstart](https://ngrok.com/docs/share-localhost/quickstart)
for current installation, authentication, and Traffic Policy details.
