sync() {
    #do things with parameters like $1 such as
    git add .
    git commit -m "$1"
    git push origin HEAD
}


syncPull() {
    #do things with parameters like $1 such as
    git reset --hard "origin/$1"
    git pull origin "$1"
}

ghash(){
    git rev-parse HEAD
}

syncHash(){
    sync "$1"
    ghash
}

export PYTHONPATH="/app"
