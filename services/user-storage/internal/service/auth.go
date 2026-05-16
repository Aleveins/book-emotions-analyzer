package service

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"golang.org/x/crypto/bcrypt"

	"github.com/Aleveins/book-emotions-analyzer/pkg/jwt"
	"github.com/Aleveins/book-emotions-analyzer/services/user-storage/internal/model"
	"github.com/Aleveins/book-emotions-analyzer/services/user-storage/internal/repository"
)

const tokenTTL = 24 * time.Hour

var (
	ErrInvalidCredentials = errors.New("invalid login or password")
	ErrInvalidLogin       = errors.New("login must be 3-64 characters")
	ErrUserAlreadyExists  = errors.New("user with this login already exists")
)

type AuthService struct {
	userRepo  *repository.UserRepository
	jwtSecret string
}

func NewAuthService(userRepo *repository.UserRepository, jwtSecret string) *AuthService {
	return &AuthService{
		userRepo:  userRepo,
		jwtSecret: jwtSecret,
	}
}

type RegisterInput struct {
	Login    string `json:"login" binding:"required,min=3,max=64"`
	Password string `json:"password" binding:"required,min=6"`
}

type LoginInput struct {
	Login    string `json:"login" binding:"required,min=3,max=64"`
	Password string `json:"password" binding:"required"`
}

type TokenResponse struct {
	Token string `json:"token"`
}

func (s *AuthService) Register(ctx context.Context, input RegisterInput) (*model.User, error) {
	input.Login = strings.TrimSpace(input.Login)
	if len(input.Login) < 3 || len(input.Login) > 64 {
		return nil, ErrInvalidLogin
	}
	hash, err := bcrypt.GenerateFromPassword([]byte(input.Password), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("hash password: %w", err)
	}

	user := &model.User{
		Login:        input.Login,
		PasswordHash: string(hash),
	}

	if err := s.userRepo.Create(ctx, user); err != nil {
		if errors.Is(err, repository.ErrUserAlreadyExists) {
			return nil, ErrUserAlreadyExists
		}
		return nil, fmt.Errorf("create user: %w", err)
	}

	return user, nil
}

func (s *AuthService) Login(ctx context.Context, input LoginInput) (*TokenResponse, error) {
	input.Login = strings.TrimSpace(input.Login)
	if len(input.Login) < 3 || len(input.Login) > 64 {
		return nil, ErrInvalidCredentials
	}
	user, err := s.userRepo.GetByLogin(ctx, input.Login)
	if err != nil {
		if errors.Is(err, repository.ErrUserNotFound) {
			return nil, ErrInvalidCredentials
		}
		return nil, fmt.Errorf("get user: %w", err)
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(input.Password)); err != nil {
		return nil, ErrInvalidCredentials
	}

	token, err := jwt.GenerateToken(user.ID, user.Login, s.jwtSecret, tokenTTL)
	if err != nil {
		return nil, fmt.Errorf("generate token: %w", err)
	}

	return &TokenResponse{Token: token}, nil
}

func (s *AuthService) GetCurrentUser(ctx context.Context, userID string) (*model.User, error) {
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		if errors.Is(err, repository.ErrUserNotFound) {
			return nil, repository.ErrUserNotFound
		}
		return nil, fmt.Errorf("get user: %w", err)
	}

	return user, nil
}
