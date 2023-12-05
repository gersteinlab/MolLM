## Dataset Generation Instructions
- You will need computing nodes for MySQL and workers. Edit the code to obtain your MySQL credentials from somewhere or hardcode them.

#### MySQL (Database) Node
1. SSH into the MySQL node
2. Run `mysql -u USERNAME -h 127.0.0.1 -pPASSWORD` to connect to the database with MySQL client
   - Run this SQL command `SET GLOBAL wait_timeout=3;` to kill connections continually to avoid having too many
   - Run this SQL command `SET GLOBAL max_connections=100000;` to allow many concurrent connections
   - This is where you can also run MySQL commands to query and update the database.
   - For future commands you must:
     - Run `USE molm_dataset;` at the start of your client session to set the default database to `molm_dataset` to avoid needing to always specify it.
       - This only sets it for the current client session
3. You can tunnel the MySQL server from port 3306 on the node to your local computer to use a graphical MySQL client like MySQL Workbench if desired
   - Run `ssh -L 3306:localhost:3306 USERNAME@MoLM_sql -N` locally where `MoLM_sql` is an SSH config you have setup in `~/.ssh/config` pointing to the node running MySQL and configured to use a ProxyJump via Grace.
   - May be useful to click through MySQL Workbench to see layout of the database tables and example data.
     - Although this is all possible via MySQL commands only too

## Preparing to spawn worker nodes
1. Ensure all previous worker nodes are killed.
   - If they are still running:
     - See "kill all worker nodes" below
2. Run the SQL command `UPDATE file_locks SET locked=0;` to unlock all files, in case they were locked from previous workers or from you during the process of killing all worker nodes.
3. The database has a table called `v2_aug_batches` where the `in_progress` column tracks whether a worker has taken a batch already to work on and the `finished` column to mark when a worker has finished a batch. Because we killed workers that were in progress, we need to mark batches that are not finished as not in progress for future workers to take them. Run `UPDATE v2_aug_batches SET in_progress=0 WHERE in_progress=1 AND finished=0;`
   - A batch is one zip file.

## Spawn worker nodes
1. Run worker/worker.py in worker nodes 
2. Progress check:
   - Use the MySQL database to see if workers are taking tasks and how many they are finishing:
     - Run `SELECT SUM(in_progress), SUM(finished) FROM v2_aug_batches;`
   - Run `CALL EstimateRemainingTime();` (and wait ~25 seconds) to see the current rate per second that the `finished` total is increasing.
   - Note each unit of progress is approximately 1/25000th of one epoch as each task involves one of the ~25,000 zip files.
     - Dividing the sum of finished by 25000 gives you an approximate count of epochs finished.

## Kill all worker nodes
1. Run the SQL command `UPDATE file_locks SET locked=1;` to lock all files
2. Wait a minute for any worker nodes to save files as needed to avoid corrupting the zips
3. Kill all nodes
