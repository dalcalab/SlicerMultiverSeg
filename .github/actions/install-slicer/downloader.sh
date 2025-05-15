if [ $1 == "Windows" ]; then
    PACK_OS="win"
elif [ $1 == "Linux" ]; then
  PACK_OS="linux"
elif [ $1 == "macOS" ]; then
  PACK_OS="macosx"
fi

PACK_ARCH="amd64"
BASE_URL="https://slicer-packages.kitware.com/api/v1"

APP_ID=$(curl -s "$BASE_URL/app?name=Slicer&limit=1" | jq -r '.[0]._id')
echo "Application id found for Slicer: $APP_ID"

RELEASE_NAME=$(curl -s "$BASE_URL/app/$APP_ID/release?sort=meta.revision&sortdir=-1" | jq -r '.[0].lowerName')
echo "Realease name found: $RELEASE_NAME"

PACK=$(curl -s "$BASE_URL/app/$APP_ID/package?release_id_or_name=$RELEASE_NAME&os=$PACK_OS&arch=$PACK_ARCH&limit=1" | jq '.[0]')
PACK_ID=$(jq -r '._id' <<< "$PACK")
PACK_NAME=$(jq -r '.name' <<< "$PACK")
echo "Name of the package found: $PACK_NAME ($PACK_ID)"

mkdir -p installer

HEADER_TMP=$(mktemp)
curl -# -D "$HEADER_TMP" "$BASE_URL/item/$PACK_ID/download" -o file
CONTENT_DISPOSITION=$(grep -i '^Content-Disposition:' "$HEADER_TMP")
FILENAME=$(echo "$CONTENT_DISPOSITION" | sed -n 's/.*filename="\(.*\)".*/\1/p')
mv file "installer/$FILENAME"

