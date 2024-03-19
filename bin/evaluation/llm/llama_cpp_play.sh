
set -x

QUERY="Please tell me a story about dragon. Here are some steps you should follow. \n1. Make an introduction about roles.\n2. Introduct some story backgrounds.\n3. Tell the story"

curl \
  --request POST \
  --url http://localhost:8080/completion \
  --header "Content-Type: application/json" \
  --data "{\"prompt\": \"${QUERY}\",\"n_predict\": 128}"
