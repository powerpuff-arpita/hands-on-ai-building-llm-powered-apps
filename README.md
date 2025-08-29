---
title: AI-LLM
emoji: :X
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: 1.46.0
pinned: false
app_file: app/app.py
license: other
---

# Hands-On AI: Building and Deploying LLM-Powered Apps
This is the repository for the LinkedIn Learning course `Hands-On AI: Building and Deploying LLM-Powered Apps`. The full course is available from [LinkedIn Learning][lil-course-url].

![lil-thumbnail-url]

Are you ready to start building applications with large language models (LLMs), but not sure where to begin? This course, which is designed uniquely for beginners with no experience in the LLM space, offers an overview of the fundamentals of LLMs with hands-on challenges to boost your skills along the way.

Explore the essentials of retrieval-augmented generation including search engine basics, embedding model limitations, and how to build a chat-with-PDF application. Along the way, instructor Han Lee shows you how to get up and running with prompt engineering, using the prompt playground for LLM apps.

This course is integrated with GitHub Codespaces, an instant cloud developer environment that offers all the functionality of your favorite IDE without the need for any local machine setup. With GitHub Codespaces, you can get hands-on practice from any machine, at any time—all while using a tool that you’ll likely encounter in the workplace. Check out the “Using GitHub Codespaces with this course” video to learn how to get started.

_See the readme file in the main branch for updated instructions and information._

## Instructions
This repository has branches for each of the videos in the course. You can use the branch pop up menu in github to switch to a specific branch and take a look at the course at that stage, or you can add `/tree/BRANCH_NAME` to the URL to go to the branch you want to access.

## Branches
The branches are structured to correspond to the videos in the course. The naming convention is `CHAPTER#_MOVIE#`. As an example, the branch named `02_03` corresponds to the second chapter and the third video in that chapter. 
Some branches will have a beginning and an end state. These are marked with the letters `b` for "beginning" and `e` for "end". The `b` branch contains the code as it is at the beginning of the movie. The `e` branch contains the code as it is at the end of the movie. The `main` branch holds the final state of the code when in the course.

When switching from one exercise files branch to the next after making changes to the files, you may get a message like this:

    error: Your local changes to the following files would be overwritten by checkout:        [files]
    Please commit your changes or stash them before you switch branches.
    Aborting

To resolve this issue:
	
    Add changes to git using this command: git add .
	Commit changes using this command: git commit -m "some message"

### Instructor

Arpita Mukherjee