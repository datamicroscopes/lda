env_path=$(basename $CONDA_ENV_PATH)
default_env=$CONDA_DEFAULT_ENV
if [ $env_path = $default_env ];  then
	exit 0
else
    `source activate $CONDA_DEFAULT_ENV`
fi